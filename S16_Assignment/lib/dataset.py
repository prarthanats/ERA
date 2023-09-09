# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:28:08 2023
@author: prarthana.ts
"""

'''

__getitem__ Method:
Retrieves an example from the dataset at the specified index.
Tokenizes the source and target text using the respective tokenizers.
Prepares encoder input, decoder input, and labels as follows:
Encoder Input: Includes special tokens [SOS] (start of sentence), [EOS] (end of sentence), and padding tokens. The source text is embedded between [SOS] and [EOS], and padding tokens are added as necessary to reach the seq_len.
Decoder Input: Includes [SOS] and padding tokens. The target text is embedded after [SOS], and padding tokens are added to match the seq_len. It does not include [EOS] because the decoder has to predict it.
Labels: Include the target text followed by [EOS] and padding tokens to reach the seq_len.
Returns a dictionary containing the following:
"encoder_input": Encoder input sequence.
"decoder_input": Decoder input sequence.
"encoder_mask": A mask indicating the positions of non-padding tokens in the encoder input.
"decoder_mask": A mask used in the decoder for causal masking, preventing it from attending to future tokens.
"label": The label sequence.
"tgt_text": The target text in its original form.
'''


import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # get a src, target pair
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

    

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
            ],
            dim = 0,
        )

        # Add only the <s>
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
            ],
            dim=0,
        )

        #
        #Doubel check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == len(enc_input_tokens) + 2
        assert decoder_input.size(0) == len(dec_input_tokens) + 1
        assert label.size(0) == len(dec_input_tokens) + 1

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
             "label" :  label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1,size, size)),diagonal=1).type(torch.int)
    return mask == 0