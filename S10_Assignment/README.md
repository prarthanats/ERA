# Application of Normalization on CIFAR 10 Dataset 

## Requirements

1. Write a customLinks to an external site. ResNet architecture for CIFAR10 that has the following architecture:
~~~
	PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
	Layer1 -
	X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
	R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
	Add(X, R1)
	Layer 2 -
	Conv 3x3 [256k]
	MaxPooling2D
	BN
	ReLU
	Layer 3 -
	X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
	R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
	Add(X, R2)
	MaxPooling with Kernel Size 4
	FC Layer 
	SoftMax
	
~~~
2. Uses One Cycle Policy such that:
~~~
	Total Epochs = 24
	Max at Epoch = 5
	LRMIN = FIND
	LRMAX = FIND
	NO Annihilation
~~~

3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
4. Batch size = 512
5. Use ADAM, and CrossEntropyLoss
6. Target Accuracy: 90%

## General Requirements

1. Using modular code
2. Collab must be importing your GitHub package, and then just running the model

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

3. Mean and Standard Deviation for the CIFAR Data is 

'mean [0.49139968 0.48215841 0.44653091]'
'standard deviation [0.24703223 0.24348513 0.26158784]'


## [Resnet Utils]('https://github.com/prarthanats/Assignment_10_Resnet_Utils')

Support wrapper repo that includes augmentations, data loader, Custom_Resnet Model, learning rate finder and training and testing code. [Resnet Utils]('https://github.com/prarthanats/Assignment_10_Resnet_Utils') 

## Notebook
The notebook for this assignment can be accessed here. It clones the Resnet Utils to run the code: [Notebook] ('https://github.com/prarthanats/ERA/blob/main/S10_Assignment/Custom_Resnet_Assignment_10.ipynb')

### Model Summary

~~~
	----------------------------------------------------------------
	        Layer (type)               Output Shape         Param #
	================================================================
	            Conv2d-1           [-1, 64, 32, 32]           1,728
	              ReLU-2           [-1, 64, 32, 32]               0
	       BatchNorm2d-3           [-1, 64, 32, 32]             128
	           Dropout-4           [-1, 64, 32, 32]               0
	         PrepBlock-5           [-1, 64, 32, 32]               0
	            Conv2d-6          [-1, 128, 32, 32]          73,728
	         MaxPool2d-7          [-1, 128, 16, 16]               0
	       BatchNorm2d-8          [-1, 128, 16, 16]             256
	              ReLU-9          [-1, 128, 16, 16]               0
	 ConvolutionBlock-10          [-1, 128, 16, 16]               0
	           Conv2d-11          [-1, 128, 16, 16]         147,456
	      BatchNorm2d-12          [-1, 128, 16, 16]             256
	             ReLU-13          [-1, 128, 16, 16]               0
	           Conv2d-14          [-1, 128, 16, 16]         147,456
	      BatchNorm2d-15          [-1, 128, 16, 16]             256
	             ReLU-16          [-1, 128, 16, 16]               0
	    ResidualBlock-17          [-1, 128, 16, 16]               0
	           Conv2d-18          [-1, 256, 16, 16]         294,912
	        MaxPool2d-19            [-1, 256, 8, 8]               0
	      BatchNorm2d-20            [-1, 256, 8, 8]             512
	             ReLU-21            [-1, 256, 8, 8]               0
	 ConvolutionBlock-22            [-1, 256, 8, 8]               0
	           Conv2d-23            [-1, 512, 8, 8]       1,179,648
	        MaxPool2d-24            [-1, 512, 4, 4]               0
	      BatchNorm2d-25            [-1, 512, 4, 4]           1,024
	             ReLU-26            [-1, 512, 4, 4]               0
	 ConvolutionBlock-27            [-1, 512, 4, 4]               0
	           Conv2d-28            [-1, 512, 4, 4]       2,359,296
	      BatchNorm2d-29            [-1, 512, 4, 4]           1,024
	             ReLU-30            [-1, 512, 4, 4]               0
	           Conv2d-31            [-1, 512, 4, 4]       2,359,296
	      BatchNorm2d-32            [-1, 512, 4, 4]           1,024
	             ReLU-33            [-1, 512, 4, 4]               0
	    ResidualBlock-34            [-1, 512, 4, 4]               0
	        MaxPool2d-35            [-1, 512, 1, 1]               0
	           Linear-36                   [-1, 10]           5,130
	================================================================
	Total params: 6,573,130
	Trainable params: 6,573,130
	Non-trainable params: 0
	----------------------------------------------------------------
	Input size (MB): 0.01
	Forward/backward pass size (MB): 8.19
	Params size (MB): 25.07
	Estimated Total Size (MB): 33.28
	----------------------------------------------------------------
~~~

### Model Graph

![model_summary](https://github.com/prarthanats/ERA/assets/32382676/5ed3cb78-6614-4662-a040-e4c82127ef86)

## Implementation and Inference Details

~~~
	Epochs - 24
	Batch Size - 512
	Number of parameters: 6,573,130 parameters
	Best Training Accuracy - 98.84% (24th Epoch)
	Best Testing Accuracy - 92.33% (24th Epoch)
	LR Scheduler: OneCycleLR with pct_start = 0.2 (~5/24) since max_lr is required at Epoch 5, out of 24 total epochs
	Optimizer - Adam Scheduler
	LR Max at 5th Epoch - 0.02656
	
~~~

### Learning Rate Range Test Curve

![lr_finder](https://github.com/prarthanats/ERA/assets/32382676/ef0b6727-a32c-4357-911b-c221fb3558bf)


### Accuracy and Loss Plots

![accuracy_loss](https://github.com/prarthanats/ERA/assets/32382676/58dfefe0-c8c9-4417-8dcc-aaa2b91b98c8)


### Misclassified Images

![missclass](https://github.com/prarthanats/ERA/assets/32382676/8f40fac8-4102-49bf-b344-defe3a99e08e)

### Training Log
~~~
EPOCH: 1
Loss=1.1014963388442993 Batch_id=97 LR=0.00351 Accuracy=46.38: 100%|██████████| 98/98 [00:19<00:00,  5.04it/s]

Test set: Average loss: 0.0029, Accuracy: 5161/10000 (51.61%)

EPOCH: 2
Loss=0.8226729035377502 Batch_id=97 LR=0.00990 Accuracy=66.64: 100%|██████████| 98/98 [00:20<00:00,  4.88it/s]

Test set: Average loss: 0.0027, Accuracy: 6142/10000 (61.42%)

EPOCH: 3
Loss=0.8532325029373169 Batch_id=97 LR=0.01780 Accuracy=70.57: 100%|██████████| 98/98 [00:19<00:00,  4.94it/s]

Test set: Average loss: 0.0020, Accuracy: 6877/10000 (68.77%)

EPOCH: 4
Loss=0.6137313842773438 Batch_id=97 LR=0.02416 Accuracy=76.64: 100%|██████████| 98/98 [00:19<00:00,  4.99it/s]

Test set: Average loss: 0.0014, Accuracy: 7837/10000 (78.37%)

EPOCH: 5
Loss=0.45450809597969055 Batch_id=97 LR=0.02656 Accuracy=81.36: 100%|██████████| 98/98 [00:19<00:00,  5.01it/s]

Test set: Average loss: 0.0011, Accuracy: 8105/10000 (81.05%)

EPOCH: 6
Loss=0.4450962543487549 Batch_id=97 LR=0.02638 Accuracy=84.77: 100%|██████████| 98/98 [00:19<00:00,  4.96it/s]

Test set: Average loss: 0.0015, Accuracy: 7709/10000 (77.09%)

EPOCH: 7
Loss=0.4443962574005127 Batch_id=97 LR=0.02583 Accuracy=86.25: 100%|██████████| 98/98 [00:19<00:00,  4.95it/s]

Test set: Average loss: 0.0010, Accuracy: 8411/10000 (84.11%)

EPOCH: 8
Loss=0.31260156631469727 Batch_id=97 LR=0.02495 Accuracy=87.65: 100%|██████████| 98/98 [00:19<00:00,  4.96it/s]

Test set: Average loss: 0.0009, Accuracy: 8551/10000 (85.51%)

EPOCH: 9
Loss=0.38763347268104553 Batch_id=97 LR=0.02375 Accuracy=89.60: 100%|██████████| 98/98 [00:19<00:00,  4.98it/s]

Test set: Average loss: 0.0009, Accuracy: 8522/10000 (85.22%)

EPOCH: 10
Loss=0.25640350580215454 Batch_id=97 LR=0.02226 Accuracy=90.79: 100%|██████████| 98/98 [00:19<00:00,  4.99it/s]

Test set: Average loss: 0.0009, Accuracy: 8568/10000 (85.68%)

EPOCH: 11
Loss=0.22346514463424683 Batch_id=97 LR=0.02053 Accuracy=91.86: 100%|██████████| 98/98 [00:19<00:00,  4.93it/s]

Test set: Average loss: 0.0008, Accuracy: 8798/10000 (87.98%)

EPOCH: 12
Loss=0.14465127885341644 Batch_id=97 LR=0.01859 Accuracy=92.82: 100%|██████████| 98/98 [00:19<00:00,  4.96it/s]

Test set: Average loss: 0.0008, Accuracy: 8819/10000 (88.19%)

EPOCH: 13
Loss=0.1668323278427124 Batch_id=97 LR=0.01652 Accuracy=93.91: 100%|██████████| 98/98 [00:19<00:00,  4.98it/s]

Test set: Average loss: 0.0008, Accuracy: 8779/10000 (87.79%)

EPOCH: 14
Loss=0.16109736263751984 Batch_id=97 LR=0.01435 Accuracy=94.43: 100%|██████████| 98/98 [00:19<00:00,  4.97it/s]

Test set: Average loss: 0.0007, Accuracy: 8937/10000 (89.37%)

EPOCH: 15
Loss=0.16804151237010956 Batch_id=97 LR=0.01216 Accuracy=95.29: 100%|██████████| 98/98 [00:19<00:00,  4.94it/s]

Test set: Average loss: 0.0007, Accuracy: 8997/10000 (89.97%)

EPOCH: 16
Loss=0.12883803248405457 Batch_id=97 LR=0.01000 Accuracy=96.29: 100%|██████████| 98/98 [00:19<00:00,  4.95it/s]

Test set: Average loss: 0.0007, Accuracy: 9060/10000 (90.60%)

EPOCH: 17
Loss=0.06172945350408554 Batch_id=97 LR=0.00793 Accuracy=97.00: 100%|██████████| 98/98 [00:19<00:00,  4.96it/s]

Test set: Average loss: 0.0006, Accuracy: 9118/10000 (91.18%)

EPOCH: 18
Loss=0.12153308838605881 Batch_id=97 LR=0.00600 Accuracy=97.48: 100%|██████████| 98/98 [00:19<00:00,  4.98it/s]

Test set: Average loss: 0.0006, Accuracy: 9152/10000 (91.52%)

EPOCH: 19
Loss=0.03876384720206261 Batch_id=97 LR=0.00427 Accuracy=97.90: 100%|██████████| 98/98 [00:19<00:00,  4.96it/s]

Test set: Average loss: 0.0006, Accuracy: 9160/10000 (91.60%)

EPOCH: 20
Loss=0.05232025310397148 Batch_id=97 LR=0.00279 Accuracy=98.30: 100%|██████████| 98/98 [00:19<00:00,  4.97it/s]

Test set: Average loss: 0.0006, Accuracy: 9206/10000 (92.06%)

EPOCH: 21
Loss=0.043127890676259995 Batch_id=97 LR=0.00159 Accuracy=98.50: 100%|██████████| 98/98 [00:19<00:00,  4.97it/s]

Test set: Average loss: 0.0006, Accuracy: 9212/10000 (92.12%)

EPOCH: 22
Loss=0.024295009672641754 Batch_id=97 LR=0.00071 Accuracy=98.67: 100%|██████████| 98/98 [00:20<00:00,  4.90it/s]

Test set: Average loss: 0.0006, Accuracy: 9217/10000 (92.17%)

EPOCH: 23
Loss=0.04970421642065048 Batch_id=97 LR=0.00018 Accuracy=98.71: 100%|██████████| 98/98 [00:19<00:00,  4.96it/s]

Test set: Average loss: 0.0006, Accuracy: 9241/10000 (92.41%)

EPOCH: 24
Loss=0.029853353276848793 Batch_id=97 LR=0.00000 Accuracy=98.84: 100%|██████████| 98/98 [00:19<00:00,  4.98it/s]

Test set: Average loss: 0.0006, Accuracy: 9233/10000 (92.33%)
~~~

