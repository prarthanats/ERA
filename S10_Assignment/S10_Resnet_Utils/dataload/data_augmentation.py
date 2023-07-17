# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:33:57 2023
@author: prarthana.ts
"""
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np


## Calculate Dataset Statistics
def dataset_statistics():

    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root='./data', train = True, download = True, transform = train_transform)
    mean = train_set.data.mean(axis=(0,1,2))/255
    std = train_set.data.std(axis=(0,1,2))/255

    return mean, std

## Train and Teset Phase transformations
def albumentation_augmentation(mean, std, config):

    train_transforms = A.Compose([
                                A.Normalize(mean = mean, std = std, always_apply = True),
                                A.PadIfNeeded(min_height=config['minHeight'], min_width=config['minWidth']),
                                A.RandomCrop(height=config['height'], width=config['width']),
                                A.HorizontalFlip(p = config['horizontalFlipProb']),
                                A.Cutout(num_holes=config['num_holes'], max_h_size=config['max_h_size'], max_w_size=config['max_w_size'], fill_value=mean),
                                A.ToGray(p = config['grayscaleProb']),
                                ToTensorV2()
                              ])

    test_transforms = A.Compose([A.Normalize(mean = mean, std = std, always_apply = True),
                                ToTensorV2()])

    return lambda img:train_transforms(image=np.array(img))["image"],lambda img:test_transforms(image=np.array(img))["image"]

