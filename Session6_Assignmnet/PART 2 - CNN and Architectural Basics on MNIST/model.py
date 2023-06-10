# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:28:34 2023
@author: prarthana.ts
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=8, kernel_size=3, padding=0) #Input:28  Output:26 RF:3
        self.bn1 = nn.BatchNorm2d(8)
       
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size=3, padding=0) #Input:26 Output:24 RF:5
        self.bn2 = nn.BatchNorm2d(16)
        
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

        self.conv6 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3) # Input:3  Output:1 RF:21
        

    def forward(self, x):
        x = F.relu(self.drop(self.conv3(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))))
        x = self.pool1(x)
        x = F.relu(self.bn6(self.conv5(F.relu(self.drop(self.bn5(self.conv4(F.relu(self.bn4(self.ts1(x))))))))))
        x = self.gp(x)
        x = self.conv6(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
    
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
# Model summary    
def model_summary(model, input_size):
    return summary(model, input_size)
