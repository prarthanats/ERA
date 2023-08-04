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

def collect_misclassified_images(self, num_images):
        misclassified_images = []
        misclassified_true_labels = []
        misclassified_predicted_labels = []
        num_collected = 0

        for batch in self.test_dataloader():
            x, y = batch
            y_hat = self.forward(x)
            pred = y_hat.argmax(dim=1, keepdim=True)
            misclassified_mask = pred.eq(y.view_as(pred)).squeeze()
            misclassified_images.extend(x[~misclassified_mask].detach())
            misclassified_true_labels.extend(y[~misclassified_mask].detach())
            misclassified_predicted_labels.extend(pred[~misclassified_mask].detach())

            num_collected += sum(~misclassified_mask)

            if num_collected >= num_images:
                break

        return misclassified_images[:num_images], misclassified_true_labels[:num_images], misclassified_predicted_labels[:num_images], len(misclassified_images)


def normalize_image(self, img_tensor):
        min_val = img_tensor.min()
        max_val = img_tensor.max()
        return (img_tensor - min_val) / (max_val - min_val)

def get_gradcam_images(self, target_layer=-1, transparency=0.5, num_images=10):
        misclassified_images, true_labels, predicted_labels, num_misclassified = self.collect_misclassified_images(num_images)
        count = 0
        k = 0
        misclassified_images_converted = list()
        gradcam_images = list()

        if target_layer == -2:
          target_layer = self.model.cpu()
        else:
          target_layer = self.convblock3_l1.cpu()

        dataset_mean, dataset_std = np.array([0.49139968, 0.48215841, 0.44653091]), np.array([0.24703223, 0.24348513, 0.26158784])
        grad_cam = GradCAM(model=self.cpu(), target_layers=target_layer, use_cuda=False)  # Move model to CPU

        for i in range(0, num_images):
            img_converted = misclassified_images[i].cpu().numpy().transpose(1, 2, 0)  # Convert tensor to numpy and transpose to (H, W, C)
            img_converted = dataset_std * img_converted + dataset_mean
            img_converted = np.clip(img_converted, 0, 1)
            misclassified_images_converted.append(img_converted)
            targets = [ClassifierOutputTarget(true_labels[i])]
            grayscale_cam = grad_cam(input_tensor=misclassified_images[i].unsqueeze(0).cpu(), targets=targets)  # Move input to CPU
            grayscale_cam = grayscale_cam[0, :]
            output = show_cam_on_image(img_converted, grayscale_cam, use_rgb=True, image_weight=transparency)
            gradcam_images.append(output)

        return gradcam_images

def create_layout(self, num_images, use_gradcam):
        num_cols = 3 if use_gradcam else 2
        fig = plt.figure(figsize=(12, 5 * num_images))
        gs = gridspec.GridSpec(num_images, num_cols, figure=fig, width_ratios=[0.3, 1, 1] if use_gradcam else [0.5, 1])

        return fig, gs

def show_images_with_labels(self, fig, gs, i, img, label_text, use_gradcam=False, gradcam_img=None):
        ax_img = fig.add_subplot(gs[i, 1])
        ax_img.imshow(img)
        ax_img.set_title("Original Image")
        ax_img.axis("off")

        if use_gradcam:
            ax_gradcam = fig.add_subplot(gs[i, 2])
            ax_gradcam.imshow(gradcam_img)
            ax_gradcam.set_title("GradCAM Image")
            ax_gradcam.axis("off")

        ax_label = fig.add_subplot(gs[i, 0])
        ax_label.text(0, 0.5, label_text, fontsize=10, verticalalignment='center')
        ax_label.axis("off")

def show_misclassified_images(self, num_images=10, use_gradcam=False, gradcam_layer=-1, transparency=0.5):
        misclassified_images, true_labels, predicted_labels, num_misclassified = self.collect_misclassified_images(num_images)

        fig, gs = self.create_layout(num_images, use_gradcam)

        if use_gradcam:
            grad_cam_images = self.get_gradcam_images(target_layer=gradcam_layer, transparency=transparency, num_images=num_images)

        for i in range(num_images):
            img = misclassified_images[i].numpy().transpose((1, 2, 0))  # Convert tensor to numpy and transpose to (H, W, C)
            img = self.normalize_image(img)  # Normalize the image

            # Show true label and predicted label on the left, and images on the right
            label_text = f"True Label: {self.classes[true_labels[i]]}\nPredicted Label: {self.classes[predicted_labels[i]]}"
            self.show_images_with_labels(fig, gs, i, img, label_text, use_gradcam, grad_cam_images[i] if use_gradcam else None)

        plt.tight_layout()
        return fig