# Application of Normalization on CIFAR 10 Dataset 

In this assignment we will be implementing the different normalization techniques such as Batch, Layer and Group on the CIFAR 10 dataset using PyTorch.

##Requirements

1. Write a customLinks to an external site. ResNet architecture for CIFAR10 that has the following architecture:
	''' 
	
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
	
	'''
2. Uses One Cycle Policy such that:
	'''
	
		Total Epochs = 24
		Max at Epoch = 5
		LRMIN = FIND
		LRMAX = FIND
		NO Annihilation
		
	'''

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

4. The dataset contains 10 classes, below are 10 sample images from each class, 

we can see that some of the classes in automobile have gray scale
Also the last image of aeroplane and bird look similar

## [Resnet Utils]('https://github.com/prarthanats/Assignment_10_Resnet_Utils')

Support wrapper repo that includes augmentations, data loader, Custom_Resnet Model, learning rate finder and training and testing code. [Resnet Utils]('https://github.com/prarthanats/Assignment_10_Resnet_Utils') 

## Notebook
The notebook for this assignment can be accessed here: 

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



## Implementation and Inference Details

~~~
	Epochs - 24
	Batch Size - 512
	Number of parameters: 6,573,130 parameters
	Best Training Accuracy - 98.84% (24th Epoch)
	Best Testing Accuracy - 92.33% (24th Epoch)
	LR Scheduler: OneCycleLR with pct_start = 0.2 (~5/24) since max_lr is required at Epoch 5, out of 24 total epochs
	Optimizer - Adam Scheduler 
	
~~~

### Learning Rate Range Test Curve


### Accuracy and Loss Plots


### Misclassified Images


### Class Level Accuracy


### Training Log
