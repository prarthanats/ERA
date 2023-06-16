# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:11:03 2023
@author: prarthana.ts
All the reusable functions
"""

import torch
from torchvision import datasets, transforms

# defining the dataloader
def train_test_dataloader(train_transformer, test_transformer, dataloader_args):
    # Train data transformations
    train_transforms = transforms.Compose(train_transformer)
    # # Test Phase transformations
    test_transforms = transforms.Compose(test_transformer)

    train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
    
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    return train_loader, test_loader


def data_statistics(train, train_loader):
    # We'd need to convert it into Numpy! Remember above we have converted it into tensors already
    train_data = train.train_data
    train_data = train.transform(train_data.numpy())

    print('[Train]')
    print(' - Numpy Shape:', train.train_data.cpu().numpy().shape)
    print(' - Tensor Shape:', train.train_data.size())
    print(' - min:', torch.min(train_data))
    print(' - max:', torch.max(train_data))
    print(' - mean:', torch.mean(train_data))
    print(' - std:', torch.std(train_data))
    print(' - var:', torch.var(train_data))

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    print(images.shape)
    print(labels.shape)

    # Let's visualize some of the images
    plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')