# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:29:54 2023
@author: prarthana.ts
"""
from model import build_transformer
from dataset import BilingualDataset, casual_mask
from config import get_config, get_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import warnings
from tqdm import tqdm
import os
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import torchmetrics
from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):

  sos_idx = tokenizer_tgt.token_to_id('[SOS]')
  eos_idx = tokenizer_tgt.token_to_id('[EOS]')

  encoder_output = model.encoder(source, source_mask)
  decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)

  while True:
    if decoder_input.size(1) == max_len:
      break

    decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)

    out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

    prob = model.predict(out[:,-1])
    _,next_word = torch.max(prob, dim=1)

    decoder_input = torch.cat(
        [decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1
    )

    if next_word == eos_idx:
      break

  return decoder_input.squeeze(0)

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], d_model=config['d_model'])

    return model

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):

    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:

        with os.popen('ssty size','r') as console:
          _, console_width = console.read().split()
          console_width = int(console_width)

    except:
        console_width =80

    with torch.no_grad():
        for batch in validation_ds:
          count += 1
          encoder_input = batch["encoder_input"].to(device)
          encoder_mask = batch["encoder_mask"].to(device)
          assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

          model_out = greedy_decode(model,encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

          source_text = batch["src_text"][0]
          target_text = batch["tgt_text"][0]

          model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

          source_texts.append(source_text)
          expected.append(target_text)
          predicted.append(model_out_text)

          print_msg('-'*console_width)
          print_msg(f"{f'SOURCE: ':>12}{source_text}")
          print_msg(f"{f'TARGET: ':>12}{target_text}")
          print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

          if count == num_examples:
            print_msg('-'*console_width)
            break

        if writer:
            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted, expected)
            writer.add_scaler('validation cer', cer,global_step)
            writer.flush()

            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted, expected)
            writer.add_scaler('validation wer', wer,global_step)
            writer.flush()

            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)
            writer.add_scaler('validation BLEU', bleu,global_step)
            writer.flush()

# run tokenizer
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds,lang):

    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def filter_item(item):
    # Filtering criteria
    if len(item['encoder_input']) <= 1 or len(item['encoder_input']) >= 150 or len(item['decoder_input']) >= len(item['encoder_input']) + 10:
        return False  # Skip this item
    return True

def pad_item(item, max_en_len, max_de_len):
    # Calculate padding tokens
    enc_padding_tokens = max_en_len - len(item['encoder_input'])
    dec_padding_tokens = max_de_len - len(item['decoder_input'])

    # Ensure non-negative padding tokens
    if enc_padding_tokens < 0 or dec_padding_tokens < 0:
        raise ValueError("Sentence is too long")

    # Perform dynamic padding
    encoder_input = torch.cat([item['encoder_input'], torch.tensor([item['pad_token']] * enc_padding_tokens, dtype=torch.int64)], dim=0)
    encoder_mask = (encoder_input != item['pad_token']).unsqueeze(0).unsqueeze(0).unsqueeze(0).int()
    label = torch.cat([item['label'], torch.tensor([item['pad_token']] * dec_padding_tokens, dtype=torch.int64)], dim=0)
    decoder_input = torch.cat([item['decoder_input'], torch.tensor([item['pad_token']] * dec_padding_tokens, dtype=torch.int64)], dim=0)
    decoder_mask = ((decoder_input != item['pad_token']).unsqueeze(0).int() & casual_mask(decoder_input.size(0))).unsqueeze(0)

    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "encoder_mask": encoder_mask,
        "decoder_mask": decoder_mask,
        "label": label,
        "src_text": item['label'],
        "tgt_text": item['tgt_text']
    }

def collate_batch(batch, train_set):
    max_en_batch_len = max(x['encoder_token_len'] for x in batch)
    max_de_batch_len = max(x['decoder_token_len'] for x in batch)

    filtered_items = [item for item in batch if filter_item(item)]
    processed_batch = [pad_item(item, max_en_batch_len, max_de_batch_len) for item in filtered_items]

    if not processed_batch:
        return {
            "encoder_input": torch.empty(0),
            "decoder_input": torch.empty(0),
            "encoder_mask": torch.empty(0),
            "decoder_mask": torch.empty(0),
            "label": torch.empty(0),
            "src_text": [],
            "tgt_text": []
        }

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



def get_ds(config):

    ds_raw = load_dataset('opus_books',f"{config['lang_src']}-{config['lang_tgt']}",split='train' )

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'])
    val_ds   = BilingualDataset(val_ds_raw, tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'])

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'],shuffle=True,collate_fn= lambda batch: collate_batch(batch, train_set=True),num_workers=16)
    val_dataloader   = DataLoader(val_ds, batch_size = 1,shuffle=True,collate_fn = lambda batch: collate_batch(batch, train_set=False),num_workers=16)

    return train_dataloader,val_dataloader, tokenizer_src, tokenizer_tgt

def train_model(config):

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("Using device:",device)

    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-6)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        print('preloaded')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc =f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:

          encoder_input = batch['encoder_input'].to(device)
          decoder_input = batch['decoder_input'].to(device)
          encoder_mask  = batch['encoder_mask'].to(device)
          decoder_mask  = batch['decoder_mask'].to(device)

          encoder_output = model.encode(encoder_input, encoder_mask)
          decoder_output = model.decode(encoder_output, encoder_mask,decoder_input, decoder_mask)
          proj_output = model.project(decoder_output)

          label = batch['label'].to(device)

          loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
          batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

          writer.add_scalar('train loss',loss.item(), global_step)
          writer.flush()

          loss.backward()

          optimizer.step()
          optimizer.zero_grad(set_to_none=True)

          global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,lambda msg: batch_iterator.write(msg),   global_step, writer)

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
         }, model_filename)

if __name__ == '__main__':
    warnings.fiterwarnings("ignore")
    config = get_config()
    train_model(config)