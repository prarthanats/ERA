# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 08:58:19 2023
@author: prarthana.ts
"""

import torch.nn as nn
import torch.nn.functional as F

dropout = 0.01

class PrepBlock(nn.Module):
    def __init__(self, dropout):
        super(PrepBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.conv(x)

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
    def forward(self, x):
        return x + self.residual(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.prep = PrepBlock(dropout)
        self.conv1 = ConvolutionBlock(64, 128)
        self.R1 = ResidualBlock(128)
        self.conv2 = ConvolutionBlock(128, 256)
        self.conv3 = ConvolutionBlock(256, 512)
        self.R2 = ResidualBlock(512)
        self.maxpool = nn.MaxPool2d(kernel_size=(4, 4))
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        x = self.prep(x)
        x = self.conv1(x)
        x = self.R1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.R2(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = x.view(-1,10)
        return F.log_softmax(x,dim=1)
        return x