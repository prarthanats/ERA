# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:45:49 2023
@author: prarthana.ts
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt

# generate model summary
def model_summary(model, input_size):
    return summary(model, input_size)
    
train_losses = []
test_losses = []
train_acc = []
test_acc = []

# train function
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
    
# test function
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
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))    

# to plot training/testing accuracy and loss
def plot_accuracy_loss():    
    t = [t_items.item() for t_items in train_losses]
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(t)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    
    
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