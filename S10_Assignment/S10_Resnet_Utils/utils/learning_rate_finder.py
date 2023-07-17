# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:37:12 2023

@author: prarthana.ts
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder


def learning_finder(model,train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    
    # Step 4: Initialize LR finder and find optimal learning rate range
    lr_finder = LRFinder(model, optimizer, criterion)
    lr_finder.range_test(train_loader, end_lr=10, num_iter=100)  # Adjust end_lr and num_iter as needed
    
    # Step 5: Plot learning rate range test
    _,lr_max = lr_finder.plot()
    # Step 6: Identify LRMIN and LRMAX based on the plot
    lr_min = lr_max / 5
    
    # Print LRMIN and LRMAX
    print("LRMIN:", lr_min)
    print("LRMAX:", lr_max)
    
    lr_finder.reset()
    
    return lr_min,lr_max