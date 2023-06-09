# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:28:04 2023
@author: prarthana.ts
"""

import torch
from torchvision import datasets, transforms

# Defining the train and test data loaders and the different transformations that need to be applied.
# Here we are converting the images to tensors and standarding the values between 0 and 1 based on the mean and standard deviation.
def data_loader(batch_size, kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)


    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader