# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:06:46 2023
@author: prarthana.ts
Data Download,Load, Transformation
"""
import torch
from torchvision import datasets, transforms

def data_loader(transform_values, dataloader_args):
  # convert data to a normalized torch.FloatTensor
  transform = transforms.Compose(transform_values)
  # choose the training and test datasets
  train_data = datasets.CIFAR10('data', train=True,
                                download=True, transform=transform)
  test_data = datasets.CIFAR10('data', train=False,
                              download=True, transform=transform)
  # train dataloader
  train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)
  # test dataloader
  test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)
  # specify the image classes
  classes = train_data.classes
  print("Unique classes of images are:", classes)
  return train_loader, test_loader,train_data, classes