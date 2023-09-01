
from model import build_transformer 
from dataset import BilingualDataset, causal_mask 
from config import get_weights_file_path 

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from pathlib import Path
 
from datasets import load_dataset 
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace 

import torchmetrics 
from torch.utils.tensorboard import SummaryWriter 
import os
from torchsummary import summary

import pytorch_lightning as pl

class PytorchLighteningTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.writer = SummaryWriter(config['experiment_name'])
        self.tokenizer_src = None
        self.tokenizer_tgt = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None

        self.training_loss =[] 

        self.count = 0 
        self.source_texts = [] 
        self.expected = [] 
        self.predicted = [] 
        self.num_examples = 1

        save_dir = "weights"
        os.makedirs(save_dir, exist_ok=True)

    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        encoder_output = self.model.encode(encoder_input, encoder_mask)
        decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = self.model.project(decoder_output)
        return proj_output

    def get_all_sentences(self, ds, lang):
        for item in ds:
            yield item['translation'][lang]

    def get_model(self, config, vocab_src_len, vocab_tgt_len):
        model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"],
                                  config["seq_len"], d_model=config["d_model"])
        return model

    def get_or_build_tokenizer(self, config, ds, lang):
        tokenizer_path = Path(config['tokenizer_file'].format(lang))
        if not Path.exists(tokenizer_path):
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
            tokenizer.train_from_iterator(self.get_all_sentences(ds, lang), trainer=trainer)
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer

    def prepare_data(self):
        config = self.config
        ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
        self.tokenizer_src = self.get_or_build_tokenizer(config, ds_raw, config['lang_src'])
        self.tokenizer_tgt = self.get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self.get_model(config, self.tokenizer_src.get_vocab_size(), self.tokenizer_tgt.get_vocab_size()).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'], eps=1e-9)

        if config['preload']: 
            model_filename = get_weights_file_path(config, config['preload']) 
            print(f'Preloading model {model_filename}') 
            state = torch.load(model_filename) 
            self.model.load_state_dict(state['model_state_dict'])
            self.trainer.global_step = state['global_step']
            print("Preloaded")

        train_ds_size = int(0.9 * len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

        self.train_ds = BilingualDataset(train_ds_raw, self.tokenizer_src, self.tokenizer_tgt, config['lang_src'],
                                         config['lang_tgt'], config['seq_len'])
        self.val_ds = BilingualDataset(val_ds_raw, self.tokenizer_src, self.tokenizer_tgt, config['lang_src'],
                                       config['lang_tgt'], config['seq_len'])

        max_len_src = 0
        max_len_tgt = 0

        for item in ds_raw:
            src_ids = self.tokenizer_src.encode(item['translation'][config['lang_src']]).ids
            tgt_ids = self.tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        print(f'Max length of source sentence: {max_len_src}')
        print(f'Max length of target sentence: {max_len_tgt}')

    def training_step(self, batch, batch_idx):
        encoder_input = batch['encoder_input']
        decoder_input = batch['decoder_input']
        encoder_mask = batch['encoder_mask']
        decoder_mask = batch['decoder_mask']
        label = batch['label']

        encoder_output = self.model.encode(encoder_input, encoder_mask)
        decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = self.model.project(decoder_output)

        loss = self.loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))

        self.log('training_loss', loss, on_epoch=True, prog_bar=True)

        self.training_loss.append(loss.item())         
        self.writer.add_scalar('training_loss', loss.item(), self.trainer.global_step) 
        self.writer.flush() 
        # Backpropagate the loss 
        loss.backward(retain_graph=True) 
            
        return loss

    def greedy_decode(self, model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device): 
    
        sos_idx = tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = tokenizer_tgt.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it for every step 
        encoder_output = model.encode(source, source_mask) 

        # Initialize the decoder input with the sos token 
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device) 

        while True: 
            if decoder_input.size(1) == max_len:  
                break 

            # build mask for target 
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device) 

            # calculate output 
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask) 

            # get next token 
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
                ], dim = 1
            )

            if next_word == eos_idx: 
                break 

        return decoder_input.squeeze(0)

    def validation_step(self, batch, batch_idx):   
        self.model.eval()
        max_len = self.config['seq_len'] 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
        if self.count == self.num_examples:         
            return 
        
        self.count += 1 
        with torch.no_grad():             
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len) 

            # check that the batch size is 1 
            assert encoder_input.size(0) == 1, "Batch  size must be 1 for val"

            model_out = self.greedy_decode(self.model, encoder_input, encoder_mask, self.tokenizer_src, self.tokenizer_tgt, max_len, device)

            source_text = batch["label"][0]
            target_text = batch["tgt_text"][0] 
            model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy()) 

            self.source_texts.append(source_text) 
            self.expected.append(target_text) 
            self.predicted.append(model_out_text) 

            source_text = self.tokenizer_src.decode(batch["encoder_input"][0].detach().cpu().numpy())
            print(f"{f'SOURCE: ':>12}{source_text}")
            print(f"{f'TARGET: ':>12}{target_text}")
            print(f"{f'PREDICTED: ':>12}{model_out_text}")  
            
            
    def on_validation_epoch_end(self):

        writer = self.writer
        if writer:

            metric = torchmetrics.CharErrorRate() 
            cer = metric(self.predicted, self.expected) 
            writer.add_scalar('validation cer', cer, self.trainer.global_step) 
            writer.flush() 

            # Compute the word error rate 
            metric = torchmetrics.WordErrorRate() 
            wer = metric(self.predicted, self.expected) 
            writer.add_scalar('validation wer', wer, self.trainer.global_step) 
            writer.flush() 

            # Compute the BLEU metric 
            metric = torchmetrics.BLEUScore() 
            bleu = metric(self.predicted, self.expected) 
            writer.add_scalar('validation BLEU', bleu, self.trainer.global_step) 
            writer.flush() 
            
        self.count = 0
        self.source_texts = [] 
        self.expected = [] 
        self.predicted = [] 

    def configure_optimizers(self):
        return self.optimizer

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config['batch_size'], shuffle=True, num_workers=4,
                          persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=4, persistent_workers=True,
                          pin_memory=True)
