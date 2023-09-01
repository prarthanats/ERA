# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:28:08 2023
@author: prarthana.ts
"""
import torch
from torch.utils.data import Dataset


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
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Tranform the text to tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add the start and end of sentence  & padding tokens
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # We will add <s> and </s>
        # We will only add  <s> and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # EOS not added for decoder

        #Make sure the number of padding tokens is not negativ. If it is , sentence too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence too long")   
        
        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [   
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0
        )    
        
        # Add only </s> token (decoder label) - </s> is never given to decoder. It has to predict it
        # Hence addding eos token in label
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, #seq_len
            "decoder_input": decoder_input, #seq_len
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1,seq_len) & (1, seq_len, seq_len)"
            "label": label,  #seq_len
            "tgt_text": tgt_text,
        }
'''
causal_mask Function:

This function generates a causal mask for the decoder to ensure that it only attends to previous positions in the sequence.
It takes size as input, which corresponds to the sequence length.
The function creates a triangular matrix where elements above the main diagonal are set to zero and others to one.
This matrix is used as a mask during decoding to enforce causality, preventing the model from attending to future tokens.

'''    
def causal_mask(size):
    # Create a causal mask
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0