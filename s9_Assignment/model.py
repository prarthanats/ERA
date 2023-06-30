# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:52:41 2023
@author: prarthana.ts
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

dropout_value = 0.01
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        #Input Block, input = 32, Output = 16, RF = 3
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (3, 3), stride = 2, padding = 1, dilation = 1, bias = False),    nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )
        
        #Covolution Block1 , input = 16, Output = 12, RF = 15, Output Channels = 64
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3, 3), stride = 1, padding = 1, dilation = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), stride = 1, padding = 1, dilation = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3, 3), stride = 1, padding = 1, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) 
        
        #Covolution Block2 , input = 16, Output = 12, RF = 15, Output Channels = 64
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3, 3), stride = 1, padding = 1, dilation = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), stride = 1, padding = 1, dilation = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3, 3), stride = 1, padding = 1, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        )
        
        
        #Covolution Block3 , input = 12, Output = 10, RF = 19, Input Channels = 128, with 64 from CB1 and 64 from CB2 concatenated
        
        self.dsb = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3, 3), padding = 0, groups = 128, bias = False),
            nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = (1, 1), padding = 0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        )
        
        #Covolution Block4 , input = 10, Output = 5, RF = 31
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=(3, 3), stride = 2, padding = 1, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=(3, 3), stride = 1, padding = 1, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        )
        
        #Output Block , input = 5 , Output = 1, RF = 47
        
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size = 5) ## Global Average Pooling
        )

        self.linear = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        
        x1 = self.conv2(x)
        x2 = self.conv3(x)
        y = torch.cat((x1,x2), 1)

        y = self.dsb(y)
        y = self.conv4(y)
        y = y + self.conv5(y)
        y = self.gap(y)
        y = y.view(y.size(0), -1)
        y = self.linear(y)
        
        y = y.view(-1, 10)
        return F.log_softmax(y, dim=-1)
    

train_losses = []
test_losses = []
train_acc = []
test_acc = []

## Get Learning Rate
def get_lr(optimizer):

    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(model, device, train_loader, train_acc, train_loss, optimizer, scheduler, criterion):

    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = criterion(y_pred, target)
        train_loss.append(loss.data.cpu().numpy().item())
        loss.backward()

        optimizer.step()
        scheduler.step()
        #lrs.append(get_lr(optimizer))

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

def test(model, device, test_loader, test_acc, test_losses, criterion):
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
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            misclassified_mask = ~pred.eq(target.view_as(pred)).squeeze()
            misclassified_images.extend(data[misclassified_mask])
            misclassified_labels.extend(target.view_as(pred)[misclassified_mask])
            misclassified_predictions.extend(pred[misclassified_mask])

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))
    
    return misclassified_images[:10], misclassified_labels[:10], misclassified_predictions[:10]