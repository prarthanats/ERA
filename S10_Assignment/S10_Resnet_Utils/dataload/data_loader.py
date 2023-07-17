# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:34:37 2023
@author: prarthana.ts
"""
from torchvision import datasets
import torch

def data_download(train_transforms, test_transforms):
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform = train_transforms)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform = test_transforms)
    classes = train_data.classes
    print("Unique classes of images are:", classes)
    return train_data, test_data, classes

def data_loading(trainset, testset, dataloader_args):
    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)
    return train_loader, test_loader