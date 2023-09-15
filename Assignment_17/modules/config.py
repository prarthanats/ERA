# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 06:52:50 2023
@author: prarthana.ts
"""
import torch


def get_bert_config():
    
    return {
        "batch_size": 1024,
        "seq_len" : 20,
        "embed_size" : 128,
        "n_heads" : 8,
        "n_code" : 8,
        "n_vocab" : 40000,
        "dropout" : 0.1,
        # n_workers : 12
        
        #optimizer
        "optim_kwargs" : {'lr':1e-4, 'weight_decay':1e-4, 'betas':(.9,.999)}
        }

def get_gpt_config():
    
    return {
        # hyperparameters
        "BATCH_SIZE" : 32,  # how many independent sequences will we process in parallel?
        "BLOCK_SIZE" : 64, # what is the maximum context length for predictions?
        "MAX_ITER" : 5000,  # number of training iterations
        "EVAL_INTER" : 500,
        "LEARNING_RATE" : 3e-4,
        "DEVICE" : "cuda" if torch.cuda.is_available() else "cpu",
        "NUM_HEAD" : 6,
        "NUM_LAYER" : 6,
        "DROPOUT" : 0.2
        }
