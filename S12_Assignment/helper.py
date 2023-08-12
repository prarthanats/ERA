# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:19:32 2023
@author: prarthana.ts
Custom PyTorch dataset with Albumentations transforms
Initialize the dataset.
Args:
    dataset (Dataset): The original dataset.
    transforms (Compose, optional): Albumentations transforms to apply. Defaults to None.
    Get a sample from the dataset at the given index
"""

import torch
from typing import Any
from albumentations import *
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import matplotlib.gridspec as gridspec
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from lightning.pytorch.tuner import Tuner


class CifarAlbumentations(Dataset):
    def __init__(self, dataset: Dataset, transforms: Compose = None) -> None:
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple:
        image, label = self.dataset[index]
        image = np.array(image)
        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        return image, label


def get_train_transforms():
    return Compose([
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        PadIfNeeded(min_height=36, min_width=36, border_mode=cv2.BORDER_REFLECT),
        RandomCrop(height=32, width=32),
        HorizontalFlip(),
        Cutout(num_holes=1, max_h_size=8, max_w_size=8, p=1.0),
        ToTensorV2()
    ])

def get_test_transforms():
    return Compose([
        Normalize(mean=[0.4914, 0.4822, 0.4471], std=[0.2469, 0.2433, 0.2615]),
        ToTensorV2()
    ])

def normalize_image(img_tensor):
        min_val = img_tensor.min()
        max_val = img_tensor.max()
        return (img_tensor - min_val) / (max_val - min_val)



