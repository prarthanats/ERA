"""
Created on Wed May 31 19:05:40 2023
@author: prarthana.ts
Main file to call the utils and net and run the entire code
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

from utils import get_trainloader,get_testloader,train,test,train_test_plot,data_visualization
from model import Net

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("CUDA Available?", use_cuda)

def mnist_main_run():   
    train_data, train_loader = get_trainloader()
    test_data, test_loader = get_testloader()
    print(train_data.data.size())
    print(train_data.targets.size())
    print(test_data.data.size())
    print(test_data.targets.size())
    
    print('Min Pixel Value: {} \nMax Pixel Value: {}'.format(train_data.data.min(), train_data.data.max()))
    print('Mean Pixel Value {} \nPixel Values Std: {}'.format(train_data.data.float().mean(), train_data.data.float().std()))
    print('Scaled Mean Pixel Value {} \nScaled Pixel Values Std: {}'.format(train_data.data.float().mean() / 255, train_data.data.float().std() / 255))
    
    print('Min Pixel Value: {} \nMax Pixel Value: {}'.format(test_data.data.min(), test_data.data.max()))
    print('Mean Pixel Value {} \nPixel Values Std: {}'.format(test_data.data.float().mean(), test_data.data.float().std()))
    print('Scaled Mean Pixel Value {} \nScaled Pixel Values Std: {}'.format(test_data.data.float().mean() / 255, test_data.data.float().std() / 255))   
    
    data_visualization(train_loader)
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
    # New Line
    criterion = F.nll_loss
    num_epochs = 20
    
    for epoch in range(1, num_epochs+1):
      print(f'Epoch {epoch}')
      train(model, device, train_loader, optimizer, criterion)
      test(model, device, test_loader, criterion)
      scheduler.step()
      
    train_test_plot()
    summary(model, input_size=(1, 28, 28))

if __name__=="__main__":
    mnist_main_run()