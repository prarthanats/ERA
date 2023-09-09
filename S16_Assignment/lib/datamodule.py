# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:28:08 2023
@author: prarthana.ts
"""


from pathlib import Path
import os
import torch

from tokenizers import Tokenizer
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import BilingualDataset
from train import get_or_build_tokenizer,causal_mask
from config import get_config
from functools import partial

class DataModuleOpusLight(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.train_data = None
        self.val_data = None
        self.tokenizer_src = None
        self.tokenizer_tgt = None

    def prepare_data(self):
        dataset_name = "opus_books"
        dataset_split = "train"
        data_config = self.config

        dataset = load_dataset(dataset_name, f"{data_config['lang_src']}-{data_config['lang_tgt']}", split=dataset_split)


    def setup(self,stage=None):
        dataset_name = "opus_books"
        dataset_split = "train"
        data_config = self.config

        dataset = load_dataset(dataset_name, f"{data_config['lang_src']}-{data_config['lang_tgt']}", split=dataset_split)

        if not self.train_data and not self.val_data:
            self.setup_tokenizers(dataset)
            self.create_train_val_datasets(dataset)

    def setup_tokenizers(self, dataset):
        self.tokenizer_src = get_or_build_tokenizer(self.config, dataset, self.config["lang_src"])
        self.tokenizer_tgt = get_or_build_tokenizer(self.config, dataset, self.config["lang_tgt"])
        self.pad_token = torch.tensor([self.tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def create_train_val_datasets(self, dataset):
        max_src_length = 150
        max_tgt_length = max_src_length + 10
        min_src_length = 5
        min_tgt_length = 5
        
        data_filtered = list(filter(lambda x: len(x['translation'][self.config['lang_src']]) <= max_src_length, dataset))
        data_filtered = [k for k in data_filtered if len(k['translation'][self.config['lang_src']]) > min_src_length]
        data_filtered = [k for k in data_filtered if len(k['translation'][self.config['lang_tgt']]) > min_tgt_length]
        data_filtered = list(filter(lambda x: len(x['translation'][self.config['lang_tgt']]) <= max_tgt_length, data_filtered))
        

        train_size = int(0.9 * len(data_filtered))
        val_size = len(data_filtered) - train_size

        train_ds, val_ds = random_split(data_filtered, [train_size, val_size])

        print(f"Training  DataSet Size : {len(train_ds)}")
        print(f"Validation DataSet Size : {len(val_ds)}")

        self.train_data = BilingualDataset(
            train_ds,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config["lang_src"],
            self.config["lang_tgt"],
            self.config["seq_len"],
        )

        self.val_data = BilingualDataset(
            val_ds,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config["lang_src"],
            self.config["lang_tgt"],
            self.config["seq_len"],
        )


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.config["batch_size"],
            num_workers=7,
            shuffle=True,
            collate_fn=partial(collate_batch,self.pad_token)
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data, 
            batch_size=1, 
            shuffle=False,
            collate_fn=partial(collate_batch,self.pad_token)
        )


def collate_batch(pad_token,bat):
    encode_batch_length = list(map(lambda x : x['encoder_input'].size(0),bat))
    decode_batch_length = list(map(lambda x : x['decoder_input'].size(0),bat))
    max_seq_len = max( encode_batch_length +  decode_batch_length)
    max_seq_len = max_seq_len + 2
    src_text = []
    tgt_text = []
    for item in bat:
        item['encoder_input'] = torch.cat([item['encoder_input'],
                                           torch.tensor([pad_token] * (max_seq_len-item['encoder_input'].size(0)), dtype = torch.int64),],dim=0)
        item['decoder_input'] = torch.cat([item['decoder_input'],
                                        torch.tensor([pad_token] * (max_seq_len-item['decoder_input'].size(0)), dtype = torch.int64),],dim=0)
        
        item['label'] = torch.cat([item['label'],
                                        torch.tensor([pad_token] * (max_seq_len-item['label'].size(0)), dtype = torch.int64),],dim=0)
    
        src_text.append(item['src_text'] )
        tgt_text.append(item['tgt_text'] )
    
    return  {'encoder_input':torch.stack([o['encoder_input'] for o in bat]), #(bs,max_seq_len)
             'decoder_input':torch.stack([o['decoder_input'] for o in bat]), #bs,max_seq_len)
             'label':torch.stack([o['label'] for o in bat]), #(bs,max_seq_len)
             "encoder_mask" : torch.stack([(o['encoder_input'] != pad_token).unsqueeze(0).unsqueeze(1).int() for o in bat]),#(bs,1,1,max_seq_len)
             "decoder_mask" : torch.stack([(o['decoder_input'] != pad_token).int() & causal_mask(o['decoder_input'].size(dim=-1)) for o
                         in bat]),
             "src_text": src_text,
             "tgt_text": tgt_text
     }
