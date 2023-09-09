# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:28:08 2023
@author: prarthana.ts
"""

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
import time

from model import build_transformer
from train import causal_mask

class TrainingModuleOpusLightning(LightningModule):
    def __init__(self, tokenizer_src,tokenizer_tgt,config):
        super().__init__()
        self.learning_rate = 1e-3
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.config = config
        self.seq_len = config["seq_len"]
        
        self.src_vocab_size = tokenizer_src.get_vocab_size()
        self.tgt_vocab_size = tokenizer_tgt.get_vocab_size()

        self.model = build_transformer(
            self.src_vocab_size,
            self.tgt_vocab_size,
            self.seq_len,
            self.seq_len,
            d_model=config["d_model"],
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1)
        
        self.training_loss = []
        self.val_loss = []
        self.source_texts = []
        self.expected = []
        self.predicted = []
        self.last_val_batch = None 
        self.epoch_loader = 0

    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        encoder_output = self.model.encode(encoder_input, encoder_mask)  # (B, seq_len, seq_len)
        decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = self.model.project(decoder_output)  # (B, seq_len, vocab_size)

        return proj_output
    
    def training_step(self, batch, batch_idx):
        self.epoch_start_time = time.time()
        encoder_input = batch["encoder_input"]
        decoder_input = batch["decoder_input"]
        encoder_mask = batch["encoder_mask"]
        decoder_mask = batch["decoder_mask"]
       
        proj_output = self.forward(encoder_input, decoder_input, encoder_mask, decoder_mask)

        label = batch["label"]  # (B, seq_len)
        loss = self.loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))
        
        # update and log metrics
        self.training_loss.append(loss)
        self.epoch_loader = 1

        return loss

    def on_train_epoch_end(self):
        mean_train_loss = sum(self.training_loss) / len(self.training_loss)
        print("TRAINING LOSS : ", round(mean_train_loss.item(), 5))
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time
        epoch_duration_minutes = epoch_duration / 60.0

        self.log("epoch_time", epoch_duration, on_epoch=True, prog_bar=True)
        print("TIME TAKEN (minutes): {:.2f}".format(epoch_duration_minutes))
        print("----------------------------------------------------------------------")
        self.train_loss = []
        

    def validation_step(self, batch, batch_idx):
        self.last_val_batch = batch
        if self.epoch_loader > 0:
            print("----------------------------------------------------------------------")
            print(f'EPOCH :  {self.current_epoch} :')
            self.model.eval()

            src = batch['encoder_input']
            src_mask = batch['encoder_mask']

            model_out = self.greedy_decode(
                src, src_mask, max_len=self.seq_len, device=self.device
            )

            source_text = batch['src_text']
            target_text = batch['tgt_text']
            model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            expected = target_text
            predicted = model_out_text
            
            print(f"SOURCE : {source_text[0]}")
            print(f"EXPECTED  : {expected[0]}")
            print(f"PREDICTED : {predicted}")

            self.model.train()
            self.epoch_loader = 0


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"], eps=1e-9)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config["lr"],
            epochs=self.trainer.max_epochs,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            div_factor=10,
            final_div_factor=10,
            three_phase=True,
            anneal_strategy='linear',
            verbose=False
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step',  
                'frequency': 1
            },
        }
    
    def greedy_decode(self, source, source_mask, max_len: int, device: str):
        sos_idx = self.tokenizer_tgt.token_to_id("[SOS]")
        eos_idx = self.tokenizer_tgt.token_to_id("[EOS]")

        # encoder output
        encoder_output = self.model.encode(source, source_mask)
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
        while True:
            if decoder_input.size(1) == max_len:
                break
            # build target mask
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            out = self.model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # get next token
            prob = self.model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
                ],
                dim=1,
            )
            if next_word == eos_idx:
                break

        return decoder_input.squeeze(0)
    
