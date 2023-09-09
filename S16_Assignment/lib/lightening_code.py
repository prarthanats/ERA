# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 11:54:12 2023
@author: prarthana.ts
"""


from model import build_transformer 
from dataset import BilingualDataset, causal_mask 
from config import get_weights_file_path 

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR

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
        self.expected = [] 
        self.predicted = [] 
        self.num_examples = 1

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

        train_ds_size = int(0.9 * len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

        self.train_ds = BilingualDataset(train_ds_raw, self.tokenizer_src, self.tokenizer_tgt, config['lang_src'],
                                         config['lang_tgt'], config['seq_len'])
        self.val_ds = BilingualDataset(val_ds_raw, self.tokenizer_src, self.tokenizer_tgt, config['lang_src'],
                                       config['lang_tgt'], config['seq_len'])
    
    
    def training_step(self, batch, batch_idx):
        encoder_input = batch['encoder_input']
        decoder_input = batch['decoder_input']
        encoder_mask  = batch['encoder_mask']
        decoder_mask  = batch['decoder_mask']

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = self.model.encode(encoder_input, encoder_mask) # (b, seq_len, d_model)
        decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = self.model.project(decoder_output) # ( b, seq_len, vocab_size)

        label = batch['label']

        loss = self.loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))
        self.training_loss.append(loss)

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True,logger=True)
        return loss

    

    def casual_mask(self, size):
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return(mask == 0)

    def greedy_decode(self,source, source_mask):

        sos_idx = self.tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = self.tokenizer_tgt.token_to_id('[EOS]')

        encoder_output = self.model.encode(source, source_mask)
        decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source)

        while True:
            if decoder_input.size(1) == self.seq_len:
              break

            decoder_mask = self.casual_mask(decoder_input.size(1)).type_as(source_mask)

            out = self.model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            prob = self.model.project(out[:,-1])
            _,next_word = torch.max(prob, dim=1)

            decoder_input = torch.cat(
                [decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item())], dim=1
            )

            if next_word == eos_idx:
              break

        return decoder_input.squeeze(0)

    def validation_step(self, batch, batch_idx):
        encoder_input = batch["encoder_input"]
        encoder_mask  = batch["encoder_mask"]


        model_out = self.greedy_decode(encoder_input, encoder_mask)

        source_text = batch["label"][0]
        target_text = batch["tgt_text"][0]


        model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

        print(f"{f'SOURCE: ':>12}{source_text}")
        print(f"{f'TARGET: ':>12}{target_text}")
        print(f"{f'PREDICTED: ':>12}{model_out_text}") 

        self.expected.append(target_text)
        self.predicted.append(model_out_text) 
            
            
    def on_validation_epoch_end(self):
        metric = torchmetrics.CharErrorRate()
        cer = metric(self.predicted, self.expected)
        self.log('validation_cer', cer, prog_bar=True, on_epoch=True, logger=True)


        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(self.predicted, self.expected)
        self.log('validation_wer', wer, prog_bar=True, on_epoch=True, logger=True)

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(self.predicted, self.expected)
        self.log('validation_bleu', bleu, prog_bar=True, on_epoch=True, logger=True)

        self.expected.clear()
        self.predicted.clear()

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], eps=1e-9)
        scheduler = OneCycleLR(
                        self.optimizer,
                        max_lr= 10**-3,
                        pct_start = 1/10,
                        epochs=self.trainer.max_epochs,
                        total_steps=self.trainer.estimated_stepping_batches,
                        div_factor=10,
                        three_phase=True,
                        final_div_factor=10,
                        anneal_strategy='linear'
                    )
        return {
             "optimizer": self.optimizer,
             "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
             }
        
    def filter_item(self, item):
        # Filtering criteria
        if len(item['encoder_input']) <= 1 or len(item['encoder_input']) >= 150 or len(
                item['decoder_input']) >= len(item['encoder_input']) + 10:
            return False  # Skip this item
        return True

    def pad_item(self, item, max_en_len, max_de_len):
        # Calculate padding tokens
        enc_padding_tokens = max_en_len - len(item['encoder_input'])
        dec_padding_tokens = max_de_len - len(item['decoder_input'])

        # Ensure non-negative padding tokens
        if enc_padding_tokens < 0 or dec_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Perform dynamic padding
        encoder_input = torch.cat(
            [item['encoder_input'], torch.tensor([item['pad_token']] * enc_padding_tokens, dtype=torch.int64)], dim=0)
        encoder_mask = (encoder_input != item['pad_token']).unsqueeze(0).unsqueeze(0).unsqueeze(0).int()
        label = torch.cat([item['label'], torch.tensor([item['pad_token']] * dec_padding_tokens, dtype=torch.int64)],
                          dim=0)
        decoder_input = torch.cat(
            [item['decoder_input'], torch.tensor([item['pad_token']] * dec_padding_tokens, dtype=torch.int64)],
            dim=0)
        decoder_mask = (
                (decoder_input != item['pad_token']).unsqueeze(0).int() & causal_mask(decoder_input.size(0))).unsqueeze(
            0)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": item['label'],
            "tgt_text": item['tgt_text']
        }

    def collate_batch(self, batch, train_set):
        max_en_batch_len = max(x['encoder_token_len'] for x in batch)
        max_de_batch_len = max(x['decoder_token_len'] for x in batch)

        filtered_items = [item for item in batch if self.filter_item(item)]
        processed_batch = [self.pad_item(item, max_en_batch_len, max_de_batch_len) for item in filtered_items]

        # Convert the generator comprehension to a list comprehension
        return {
            "encoder_input": torch.vstack([item["encoder_input"] for item in processed_batch]),
            "decoder_input": torch.vstack([item["decoder_input"] for item in processed_batch]),
            "encoder_mask": torch.vstack([item["encoder_mask"] for item in processed_batch]),
            "decoder_mask": torch.vstack([item["decoder_mask"] for item in processed_batch]),
            "label": torch.vstack([item["label"] for item in processed_batch]),
            "src_text": [item["label"] for item in processed_batch],
            "tgt_text": [item["tgt_text"] for item in processed_batch]
        }

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config['batch_size'], shuffle=True, num_workers=4,
                      collate_fn=lambda batch: self.collate_batch(batch, train_set=True), persistent_workers=True,
                      pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=4,
                      collate_fn=lambda batch: self.collate_batch(batch, train_set=False), persistent_workers=True,
                      pin_memory=True)


