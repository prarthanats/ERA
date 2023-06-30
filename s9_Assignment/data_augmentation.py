# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:24:29 2023
@author: prarthana.ts
"""


from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

def normalize_calculation(dataset):
  train_transform = transforms.Compose([transforms.ToTensor()])
  train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
  mean = train_set.data.mean(axis=(0,1,2))/255
  std = train_set.data.std(axis=(0,1,2))/255
  return mean, std


def data_augmentation(mean, std):

    train_transforms = A.Compose([
                                  A.HorizontalFlip(),
                                  A.ShiftScaleRotate(),
                                  A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, 
                                                  min_height=16, min_width=16,
                                                  fill_value = tuple([x * 255.0 for x in mean]),
                                                  mask_fill_value=None),
                                  A.Normalize(mean = mean, std = std, always_apply = True),
                                  ToTensorV2()

                              ])

    test_transforms = A.Compose([A.Normalize(mean = mean, std = std, always_apply = True),
                                ToTensorV2()])

    return lambda img:train_transforms(image=np.array(img))["image"],lambda img:test_transforms(image=np.array(img))["image"]
