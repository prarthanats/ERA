# Application of CAMs, LRs and Optimizers on CIFAR 10 Dataset 

In this assignment we will be implementing a ResNet18 architecture on the CIFAR 10 dataset using PyTorch. 

## Requirements

1. Write a customLinks to an external site. ResNet18 architecture for CIFAR10 that has the following architecture from [Resnet18](https://github.com/kuangliu/pytorch-cifar)
2. Transforms while training: RandomCrop(32, padding=4),CutOut(16x16)
3. Train for 20 epochs
	
## General Requirements

1. Using modular code
2. Collab must be importing your GitHub package, and then just running the model
3. Get 10 misclassified images
4. Get 10 GradCam outputs on any misclassified images (remember that you MUST use the library we discussed in the class)

## CIFAR Data
The CIFAR-10 dataset consists of 60000 32x32 RGB colour images  each of size 32x32 pixels, in 10 classes. There are 50000 training images and 10000 test images. Analysis on the dataset can be found here. 

1. Images are equally distributed across classes, no class imbalance
2. The 10 classes in CIFAR-10 are:

	Airplane
	Automobile
	Bird
	Cat
	Deer
	Dog
	Frog
	Horse
	Ship
	Truck	

3. Mean and Standard Deviation for the CIFAR Data .
~~~
	Mean is '0.49139968, 0.48215841, 0.44653091'
	standard deviation is '0.24703223, 0.24348513, 0.26158784'
~~~

## [Torch Wrapper]('https://github.com/prarthanats/torch_wrapper.git')

Support wrapper repo that includes augmentations, data loader, Custom_Resnet Model, learning rate finder and training and testing code. It has the following folder structure
~~~
    Torch Wrapper
    |──config
    |── ── assignment_10.yaml
    |── ── assignment_11.yaml
    |── model
    |── ── custom_resnet.py
    |── ── resnet.py
    |── utils
    |── ── data_augmentation.py
    |── ── data_handeling.py
    |── ── data_loader.py
    |── ── gradcam.py
    |── ── helper.py
    |── ── train_test.py
    |── ── visulaization.py
    |── main.py
    |── README.md

~~~

## Notebook
The notebook for this assignment can be accessed here: [Assignment_11_CIFAR_10](https://github.com/prarthanats/ERA/blob/main/S11_Assignment/CIFAR_10_Assignment_11.ipynb)


### Model Summary

~~~
	----------------------------------------------------------------
	        Layer (type)               Output Shape         Param #
	================================================================
	            Conv2d-1           [-1, 64, 32, 32]           1,728
	       BatchNorm2d-2           [-1, 64, 32, 32]             128
	            Conv2d-3           [-1, 64, 32, 32]          36,864
	       BatchNorm2d-4           [-1, 64, 32, 32]             128
	            Conv2d-5           [-1, 64, 32, 32]          36,864
	       BatchNorm2d-6           [-1, 64, 32, 32]             128
	        BasicBlock-7           [-1, 64, 32, 32]               0
	            Conv2d-8           [-1, 64, 32, 32]          36,864
	       BatchNorm2d-9           [-1, 64, 32, 32]             128
	           Conv2d-10           [-1, 64, 32, 32]          36,864
	      BatchNorm2d-11           [-1, 64, 32, 32]             128
	       BasicBlock-12           [-1, 64, 32, 32]               0
	           Conv2d-13          [-1, 128, 16, 16]          73,728
	      BatchNorm2d-14          [-1, 128, 16, 16]             256
	           Conv2d-15          [-1, 128, 16, 16]         147,456
	      BatchNorm2d-16          [-1, 128, 16, 16]             256
	           Conv2d-17          [-1, 128, 16, 16]           8,192
	      BatchNorm2d-18          [-1, 128, 16, 16]             256
	       BasicBlock-19          [-1, 128, 16, 16]               0
	           Conv2d-20          [-1, 128, 16, 16]         147,456
	      BatchNorm2d-21          [-1, 128, 16, 16]             256
	           Conv2d-22          [-1, 128, 16, 16]         147,456
	      BatchNorm2d-23          [-1, 128, 16, 16]             256
	       BasicBlock-24          [-1, 128, 16, 16]               0
	           Conv2d-25            [-1, 256, 8, 8]         294,912
	      BatchNorm2d-26            [-1, 256, 8, 8]             512
	           Conv2d-27            [-1, 256, 8, 8]         589,824
	      BatchNorm2d-28            [-1, 256, 8, 8]             512
	           Conv2d-29            [-1, 256, 8, 8]          32,768
	      BatchNorm2d-30            [-1, 256, 8, 8]             512
	       BasicBlock-31            [-1, 256, 8, 8]               0
	           Conv2d-32            [-1, 256, 8, 8]         589,824
	      BatchNorm2d-33            [-1, 256, 8, 8]             512
	           Conv2d-34            [-1, 256, 8, 8]         589,824
	      BatchNorm2d-35            [-1, 256, 8, 8]             512
	       BasicBlock-36            [-1, 256, 8, 8]               0
	           Conv2d-37            [-1, 512, 4, 4]       1,179,648
	      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
	           Conv2d-39            [-1, 512, 4, 4]       2,359,296
	      BatchNorm2d-40            [-1, 512, 4, 4]           1,024
	           Conv2d-41            [-1, 512, 4, 4]         131,072
	      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
	       BasicBlock-43            [-1, 512, 4, 4]               0
	           Conv2d-44            [-1, 512, 4, 4]       2,359,296
	      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
	           Conv2d-46            [-1, 512, 4, 4]       2,359,296
	      BatchNorm2d-47            [-1, 512, 4, 4]           1,024
	       BasicBlock-48            [-1, 512, 4, 4]               0
	           Linear-49                   [-1, 10]           5,130
	================================================================
	Total params: 11,173,962
	Trainable params: 11,173,962
	Non-trainable params: 0
	----------------------------------------------------------------
	Input size (MB): 0.01
	Forward/backward pass size (MB): 11.25
	Params size (MB): 42.63
	Estimated Total Size (MB): 53.89
	----------------------------------------------------------------
~~~

## Implementation and Inference Details

~~~
	Epochs - 20
	Batch Size - 512
	Number of parameters:11,173,962 parameters
	Best Training Accuracy - 96.17% (20th Epoch)
	Best Testing Accuracy - 91.51% (20th Epoch)	
~~~

### Training Logs
~~~
	Epoch 1:
	Loss=1.5638378858566284 Batch_id=97 LR=0.00464 Accuracy=35.93: 100%|██████████| 98/98 [00:41<00:00,  2.37it/s]
	
	Test set: Average loss: 0.0036, Accuracy: 3916/10000 (39.16%)
	
	Epoch 2:
	Loss=1.4184805154800415 Batch_id=97 LR=0.01309 Accuracy=49.66: 100%|██████████| 98/98 [00:42<00:00,  2.33it/s]
	
	Test set: Average loss: 0.0045, Accuracy: 4029/10000 (40.29%)
	
	Epoch 3:
	Loss=1.1353524923324585 Batch_id=97 LR=0.02353 Accuracy=58.18: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0036, Accuracy: 5029/10000 (50.29%)
	
	Epoch 4:
	Loss=0.9001946449279785 Batch_id=97 LR=0.03194 Accuracy=65.84: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0016, Accuracy: 7170/10000 (71.70%)
	
	Epoch 5:
	Loss=0.8339181542396545 Batch_id=97 LR=0.03511 Accuracy=71.57: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0016, Accuracy: 7339/10000 (73.39%)
	
	Epoch 6:
	Loss=0.7122399806976318 Batch_id=97 LR=0.03472 Accuracy=75.86: 100%|██████████| 98/98 [00:42<00:00,  2.33it/s]
	
	Test set: Average loss: 0.0013, Accuracy: 7720/10000 (77.20%)
	
	Epoch 7:
	Loss=0.6160628199577332 Batch_id=97 LR=0.03358 Accuracy=78.90: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0018, Accuracy: 7401/10000 (74.01%)
	
	Epoch 8:
	Loss=0.4498029351234436 Batch_id=97 LR=0.03174 Accuracy=81.35: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0010, Accuracy: 8250/10000 (82.50%)
	
	Epoch 9:
	Loss=0.5891032218933105 Batch_id=97 LR=0.02928 Accuracy=83.49: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0011, Accuracy: 8218/10000 (82.18%)
	
	Epoch 10:
	Loss=0.3976404070854187 Batch_id=97 LR=0.02630 Accuracy=85.02: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0009, Accuracy: 8540/10000 (85.40%)
	
	Epoch 11:
	Loss=0.36522236466407776 Batch_id=97 LR=0.02295 Accuracy=87.00: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0008, Accuracy: 8606/10000 (86.06%)
	
	Epoch 12:
	Loss=0.3309040069580078 Batch_id=97 LR=0.01935 Accuracy=88.17: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0009, Accuracy: 8609/10000 (86.09%)
	
	Epoch 13:
	Loss=0.3672555387020111 Batch_id=97 LR=0.01568 Accuracy=89.78: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0008, Accuracy: 8799/10000 (87.99%)
	
	Epoch 14:
	Loss=0.24744948744773865 Batch_id=97 LR=0.01210 Accuracy=90.92: 100%|██████████| 98/98 [00:42<00:00,  2.33it/s]
	
	Test set: Average loss: 0.0007, Accuracy: 8924/10000 (89.24%)
	
	Epoch 15:
	Loss=0.21767078340053558 Batch_id=97 LR=0.00875 Accuracy=92.20: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0007, Accuracy: 8970/10000 (89.70%)
	
	Epoch 16:
	Loss=0.20082253217697144 Batch_id=97 LR=0.00578 Accuracy=93.40: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0006, Accuracy: 9076/10000 (90.76%)
	
	Epoch 17:
	Loss=0.12145695090293884 Batch_id=97 LR=0.00333 Accuracy=94.75: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0006, Accuracy: 9117/10000 (91.17%)
	
	Epoch 18:
	Loss=0.13016997277736664 Batch_id=97 LR=0.00150 Accuracy=95.48: 100%|██████████| 98/98 [00:42<00:00,  2.33it/s]
	
	Test set: Average loss: 0.0006, Accuracy: 9123/10000 (91.23%)
	
	Epoch 19:
	Loss=0.0886826142668724 Batch_id=97 LR=0.00038 Accuracy=95.89: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0006, Accuracy: 9150/10000 (91.50%)
	
	Epoch 20:
	Loss=0.12358154356479645 Batch_id=97 LR=0.00000 Accuracy=96.17: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0006, Accuracy: 9151/10000 (91.51%)
~~~

### Learning Rate Range Test Curve

![LR](https://github.com/prarthanats/ERA/assets/32382676/878497b5-1042-4aca-80ad-c7c6d892fbaf)

### Accuracy and Loss Plots

![Loss and Accuracy](https://github.com/prarthanats/ERA/assets/32382676/ce3705ef-991b-41cf-b514-1e1c3b67950f)

### Misclassified Images

![Misclassify](https://github.com/prarthanats/ERA/assets/32382676/a223abee-b913-4272-b989-6087e96e4ca2)

### GradCam Misclassified Images

![GradCam](https://github.com/prarthanats/ERA/assets/32382676/d3c84aed-312c-4c8b-9d20-58a79d9690e2)

### TensorBoard Outputs

<img width="626" alt="tensor Output" src="https://github.com/prarthanats/ERA/assets/32382676/7ebdf3c1-5a7b-4417-bece-681ca114fdc5">


### Training Log
