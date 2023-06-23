# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 19:44:52 2023
@author: prarthana.ts
"""

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary

train_losses = []
test_losses = []
train_acc = []
test_acc = []
train_loss = []
train_accuracy = []

dropout_value = 0.01

def norm_layer(norm, parameter):
    if norm == "BN":
        return(nn.BatchNorm2d(parameter[0]))
    elif norm == "LN":
        return(nn.GroupNorm(1,parameter[0]))
    elif norm == "GN":
        return nn.GroupNorm(2, parameter[0])
    else:
        raise ValueError('Options are BN / LN / GN')


class Net(nn.Module):
    def __init__(self, norm="BN"):
        super(Net, self).__init__()
        self.norm = norm
        print(norm)
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1,bias = False),
            norm_layer(self.norm, [8, 32, 32]), 
            nn.Dropout(dropout_value),
            nn.ReLU()
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1,bias = False),
            norm_layer(self.norm, [16, 32, 32]),
            nn.Dropout(dropout_value), 
            nn.ReLU()
        )
        self.tr1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1,bias = False),
            norm_layer(self.norm, [16, 16, 16]), 
            nn.Dropout(dropout_value), 
            nn.ReLU()
        )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1,bias = False),
            norm_layer(self.norm, [32, 16, 16]), 
            nn.Dropout(dropout_value),
            nn.ReLU()
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1,bias = False),
            norm_layer(self.norm, [32, 16, 16]), 
            nn.Dropout(dropout_value),
            nn.ReLU()
        )
       
        self.tr2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0,bias = False),
            norm_layer(self.norm, [16, 8, 6]), 
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) 
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0,bias = False),
            norm_layer(self.norm, [32, 6, 4]), 
            nn.Dropout(dropout_value),
            nn.ReLU()
        )
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0,bias = False),
            norm_layer(self.norm, [64, 4, 2]), 
            nn.Dropout(dropout_value),
            nn.ReLU()
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=2)
        )
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout_value)
   
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        
        x = self.tr1(x)
        x = self.pool1(x)
        
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        
        x = self.tr2(x)
        x = self.pool2(x)

        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)

        x = self.gap(x)
        x = self.convblock10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
        # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)
    
    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

  train_accuracy.append(train_acc[-1])
  train_loss.append([x.item() for x in train_losses][-1])
  

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    misclassified_images = []
    misclassified_labels = []
    misclassified_predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Check for misclassified images
            misclassified_mask = ~pred.eq(target.view_as(pred)).squeeze()
            misclassified_images.extend(data[misclassified_mask])
            misclassified_labels.extend(target.view_as(pred)[misclassified_mask])
            misclassified_predictions.extend(pred[misclassified_mask])

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_acc.append(100. * correct / len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return misclassified_images[:10], misclassified_labels[:10], misclassified_predictions[:10]
    
    
class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        #28 > 28 |3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1,bias = False),
            nn.ReLU()
        )
        #28 >28 |5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1,bias = False),
            nn.ReLU()
        )
       #28 >14 |6
        self.pool1 = nn.MaxPool2d(2, 2)
       #14 > 14 |6 
        self.tr1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        #14 >12 |10
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0,bias = False),
            nn.ReLU(),
        )
        #12> 10 | 14
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0,bias = False),
            nn.ReLU(),
        )
        #10 > 10 | 14
        self.tr2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        ) 
        #10 > 8 | 18
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0,bias = False),
            nn.ReLU()
        )
        #8 > 6 |22
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0,bias = False),
            nn.ReLU()
        )
        #6 > 1 |32
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(6, 6), padding=0,bias = False),
        )


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.tr1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.tr2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
class model2(nn.Module):
    def __init__(self):
        super(model2, self).__init__()
        # Input Block
        #28 > 28 |3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1,bias = False),
            nn.ReLU()
        )

        # CONVOLUTION BLOCK 1  #28 >28 |5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=1,bias = False),
            nn.ReLU()
        )
        #28 >14 |6
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        #14 > 14 |6 
        self.tr1 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        #14 >12 |10
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0,bias = False),
            nn.ReLU()
        )
        #12> 10 | 14
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0,bias = False),
            nn.ReLU()
        )
        #10 > 10 | 14
        self.tr2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        #10 > 8 | 18
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0,bias = False),
            nn.ReLU()
        )
        #8 > 6 |22
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0,bias = False),
            nn.ReLU()
        )
        #6 > 1 |32
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(6, 6), padding=0,bias = False),
        )


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.tr1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.tr2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
class model3(nn.Module):
    def __init__(self):
        super(model3, self).__init__()
        # Input Block #28 > 28 |3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1,bias = False),
            nn.ReLU()
        )

        # CONVOLUTION BLOCK 1 #28 >28 |5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=1,bias = False),
            nn.ReLU()
        )
        #28 >14 |6
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        #14 > 14 |6 
        self.tr1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        #14 >12 |10
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0,bias = False),
            nn.ReLU()

        )
        #12> 10 | 14
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0,bias = False),
            nn.ReLU()
        )
        #10 > 10 | 14
        self.tr2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        #10 > 8 | 18
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0,bias = False),
            nn.ReLU()
        )
        #8 > 6 |22
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0,bias = False),
            nn.ReLU()
        )
        #6 > 1 |32
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(6, 6), padding=0,bias = False),
        )


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.tr1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.tr2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
class model4(nn.Module):
    def __init__(self):
        super(model4, self).__init__()
        # Input Block #28 > 28 |3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1,bias = False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )
        # CONVOLUTION BLOCK 1 #28 >28 |5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=1,bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        #28 >14 |6
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        #14 > 14 |6 
        self.tr1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        #14 >12 |10
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )
        #12> 10 | 14
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU()

        )
        #10 > 10 | 14
        self.tr2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=8, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        #10 > 8 | 18
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(10),
			nn.ReLU()
        )
        #8 > 6 |22
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(10),
        )
        #6 > 1 |32
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(6, 6), padding=0,bias = False),
        )


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.tr1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.tr2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
class model5(nn.Module):
    def __init__(self):
        super(model5, self).__init__()
        # Input Block #28 > 28 |3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1,bias = False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )

        # CONVOLUTION BLOCK 1 #28 >28 |5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=1,bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        #28 >14 |6
        self.pool1 = nn.MaxPool2d(2, 2)
        #14 > 14 |6 
        self.tr1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        #14 >12 |10
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )
        #12> 10 | 14
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        #10 > 10 | 14
        self.tr2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=8, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        #10 > 8 | 18
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0,bias = False),
            nn.ReLU()
        )
        #8 > 6 |22
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(10),
        )
        #6 > 1 |32
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(6, 6), padding=0,bias = False),
        )
        self.dropout = nn.Dropout(0.01)


    def forward(self, x):
        x = self.convblock1(x)
        x = self.dropout(x)
        x = self.convblock2(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.tr1(x)
        x = self.convblock3(x)
        x = self.dropout(x)
        x = self.convblock4(x)
        x = self.dropout(x)
        x = self.tr2(x)
        x = self.convblock5(x)
        x = self.dropout(x)
        x = self.convblock6(x)
        x = self.dropout(x)
        x = self.convblock7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
class model6(nn.Module):
    def __init__(self):
        super(model6, self).__init__()
        # Input Block #28 > 28 |3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1,bias = False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )

        # CONVOLUTION BLOCK 1 #28 >28 |5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=1,bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        #28 >14 |6
        self.pool1 = nn.MaxPool2d(2, 2)
        #14 > 14 |6 
        self.tr1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        #14 >12 |10
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )
        #12> 10 | 14
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        #10 > 10 | 14
        self.tr2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=8, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        #10 > 8 | 18
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0,bias = False),
            nn.ReLU()
        )
        #8 > 6 |22
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(10),
        )
        #6 > 1 |32
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )
        #1 > 1 |32
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0,bias = False),
        )
        self.dropout = nn.Dropout(0.01)


    def forward(self, x):
        x = self.convblock1(x)
        x = self.dropout(x)
        x = self.convblock2(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.tr1(x)
        x = self.convblock3(x)
        x = self.dropout(x)
        x = self.convblock4(x)
        x = self.dropout(x)
        x = self.tr2(x)
        x = self.convblock5(x)
        x = self.dropout(x)
        x = self.convblock6(x)
        x = self.dropout(x)
        x = self.gap(x)
        x = self.convblock7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
class model7(nn.Module):
    def __init__(self):
        super(model7, self).__init__()
        # Input Block #28 > 28 |3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1,bias = False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )

        # CONVOLUTION BLOCK 1 #28 >28 |5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=1,bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        #28 >14 |6
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        #14 > 14 |6 
        self.tr1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        #14 >12 |10
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        #12> 10 | 14
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        #10 > 10 | 14
        self.tr2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        #10 > 8 | 18
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        #8 > 6 |22
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        #6 > 1 |32
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )
        #1 > 1 |32
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0,bias = False),
        )
        self.dropout = nn.Dropout(0.01)


    def forward(self, x):
        x = self.convblock1(x)
        x = self.dropout(x)
        x = self.convblock2(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.tr1(x)
        x = self.convblock3(x)
        x = self.dropout(x)
        x = self.convblock4(x)
        x = self.dropout(x)
        x = self.tr2(x)
        x = self.convblock5(x)
        x = self.dropout(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = self.convblock7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
class model10(nn.Module):
    def __init__(self):
        super(model10, self).__init__()
        # Input Block #28 > 28 |3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1,bias = False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 28

        # CONVOLUTION BLOCK 1 #28 >28 |5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=1,bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU()
        ) # output_size = 28
        #28 >14 |6
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        #14 > 14 |6 
        self.tr1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        #14 >12 |10
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        #12> 10 | 14
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        #10 > 10 | 14
        self.tr2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0,bias = False),
            nn.ReLU()
        )
        #10 > 8 | 18
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )
        #8 > 6 |22
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0,bias = False),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        #6 > 1 |32
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )
        #1 > 1 |32
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0,bias = False),
        )
        self.dropout = nn.Dropout(0.01)


    def forward(self, x):
        x = self.convblock1(x)
        x = self.dropout(x)
        x = self.convblock2(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.tr1(x)
        x = self.convblock3(x)
        x = self.dropout(x)
        x = self.convblock4(x)
        x = self.dropout(x)
        x = self.tr2(x)
        x = self.convblock5(x)
        x = self.dropout(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = self.convblock7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
        

class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=8, kernel_size=3, padding=0) #Input:28  Output:26 RF:3
        self.bn1 = nn.BatchNorm2d(8)
        self.drop = nn.Dropout(0.05)
       
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size=3, padding=0) #Input:26 Output:24 RF:5
        self.bn2 = nn.BatchNorm2d(16)
        self.drop = nn.Dropout(0.05)
        
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size=3, padding=0) # Input:24  Output:22 RF:7
        self.bn3 = nn.BatchNorm2d(32)
        self.drop = nn.Dropout(0.05)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2) #Input:22  Output:11 RF:14
        self.ts1 = nn.Conv2d(in_channels = 32, out_channels=8, kernel_size=1, padding=0)# Input:11  Output:11 RF:14
        self.bn4 = nn.BatchNorm2d(8)
        
        self.conv4 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size=3, padding=0)# Input:11  Output:9 RF:16
        self.bn5 = nn.BatchNorm2d(16)
        self.drop = nn.Dropout(0.05)
        
        self.conv5 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size=3, padding=0)# Input:9  Output:7 RF:18
        self.bn6 = nn.BatchNorm2d(32)
                
        self.gp = nn.AvgPool2d(kernel_size=2) # Input: 7  Output:3 RF:18

        self.conv6 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3) # Input:9  Output:1 RF:21
        

    def forward(self, x):
        x = F.relu(self.drop(self.bn3(self.conv3(F.relu(self.drop(self.bn2(self.conv2(F.relu(self.drop(self.bn1(self.conv1(x))))))))))))
        x = self.pool1(x)
        x = F.relu(self.bn6(self.conv5(F.relu(self.drop(self.bn5(self.conv4(F.relu(self.bn4(self.ts1(x))))))))))
        x = self.gp(x)
        x = self.conv6(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
    
