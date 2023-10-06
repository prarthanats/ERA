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

def get_vit_config():
    
    return{
        "img_size":224, # Training resolution from Table 3 in ViT paper
        "in_channels":3, # Number of channels in input image
        "patch_size":16, # Patch size
        "num_transformer_layers":12, # Layers from Table 1 for ViT-Base
        "embedding_dim":768, # Hidden size D from Table 1 for ViT-Base
        "mlp_size":3072, # MLP size from Table 1 for ViT-Base
        "num_heads":12, # Heads from Table 1 for ViT-Base
        "attn_dropout":0, # Dropout for attention projection
        "mlp_dropout":0.1, # Dropout for dense/MLP layers 
        "embedding_dropout":0.1, # Dropout for patch and position embeddings
        "num_classes":1000
        }