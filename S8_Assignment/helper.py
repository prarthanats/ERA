# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:16:34 2023
@author: prarthana.ts
"""
from torchvision import datasets, transforms
from torchsummary import summary

def mean_std_calculation(dataset):
  train_transform = transforms.Compose([transforms.ToTensor()])
  train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
  mean = train_set.data.mean(axis=(0,1,2))/255
  std = train_set.data.std(axis=(0,1,2))/255
  return mean, std
    
def model_summary(model, input_size):
    summary(model, input_size)