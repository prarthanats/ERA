# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:29:54 2023
@author: prarthana.ts
"""


from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


'''
Greedy Decode Function (greedy_decode):

This function takes the trained model, source input, source mask, tokenizers for source and target languages, maximum sequence length (max_len), and device as input.
It performs greedy decoding to generate a translation from the given source input.
The function iteratively predicts the next token and appends it to the decoder input until an end-of-sequence token ([EOS]) is generated or the maximum sequence length is reached.
It returns the decoded sequence.

'''
def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it or every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=0,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
    
'''
Data Preprocessing Functions (get_all_sentences and get_or_build_tokenizer):

get_all_sentences: Iterates through a dataset and yields sentences in a specified language.
get_or_build_tokenizer: Loads or builds a tokenizer for a language and returns it.
'''


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    if not Path.exists(tokenizer_path):
        # code inspired from huggingface tokenizers
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def collate_batch(pad_token,bat):
    encode_batch_length = list(map(lambda x : x['encoder_input'].size(0),bat))
    decode_batch_length = list(map(lambda x : x['decoder_input'].size(0),bat))
    max_seq_len = max(max(encode_batch_length), max(decode_batch_length)) + 2
    
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
