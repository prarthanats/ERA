# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:19:32 2023
@author: prarthana.ts
"""

import torch
import yaml

def load_config_variables(file_name):
    with open(file_name, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
            print(" Loading config ..")
            #globals().update(config)
            print(" Config succesfully loaded ")
            return config
        except ValueError:
            print("Invalid yaml file")
            exit(-1)
            
from torchsummary import summary

def model_summary(model, input_size):
    summary(model, input_size)
    
         
def gpu_check(seed_value = 1):
    
    ##Set seed for reproducibility
    SEED = seed_value

    # CUDA?
    cuda = torch.cuda.is_available()
    if cuda:
        print("CUDA is available")
    else:
        print("CUDA unavailable")

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)
    
    device = torch.device("cuda" if cuda else "cpu")
    
    return device, cuda