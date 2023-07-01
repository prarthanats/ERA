#  Advanced Convolutions, Data Augmentation and Visualization on CIFAR 10 Dataset 

In this assignment we will be implementing the different Convolutions such as Dilated Convolution, Depth-wise Seperable Convolutions and Skip Connection, Data Augmentation and Visualization on the CIFAR 10 dataset using PyTorch.

## Requirements

1. Write a new network that has the architecture to C1C2C3C40 
3. Total RF must be more than 44
4. One of the layers must use Depthwise Separable Convolution
5. One of the layers must use Dilated Convolution
6. Use GAP (compulsory):- add FC after GAP to target #of classes
7. Use albumentation library and apply:horizontal flip,shiftScaleRotate, coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
8. Achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k

## General Requirements

1. Using modular code

## CIFAR Data
The CIFAR-10 dataset consists of 60000 32x32 RGB colour images  each of size 32x32 pixels, in 10 classes. There are 50000 training images and 10000 test images. Analysis on the dataset can be found here. 

1. Images are equally distributed across classes, no class imbalance
2. There are  10 classes in CIFAR-10, we can see that some of the classes in automobile have gray scale and also the last image of aeroplane and bird look similar

![Data_Analysis](https://github.com/prarthanats/ERA/assets/32382676/6c7557d3-379c-470b-808a-e87b0156c93b)

3.Mean and Standard Deviation for the CIFAR Data is $0.49139968, 0.48215841, 0.44653091$ and $0.24703223, 0.24348513, 0.26158784$

## Dilated convolution

Dilated convolution is just a convolution applied to input with defined gaps. With this definition, for given input is an 2D image, dilation rate k=1 is normal convolution and k=2 means skipping one pixel per input and k=4 means skipping 3 pixels

Dilation convolution is commonly used within the convolutional blocks to capture multi-scale information and increase the receptive field of the network. In this case, dilation convolution is applied alongside traditional convolutions within the convolution block to extract features at different scales. By adjusting the dilation rate, the receptive field of each convolutional layer can be expanded without increasing the number of parameters or reducing spatial resolution.

![dilated convolution](https://github.com/prarthanats/ERA/assets/32382676/0678183b-9e87-4218-89e9-e0b8d39bb760)

### Receptive Field wrt to Dilated Convolution

The receptive field of a neuron refers to the spatial extent of the input that influences the neuron's output. In traditional convolutions, each neuron's receptive field is determined by the size of the kernel and the stride of the convolutional layers. However, in dilated convolutions, an additional parameter called dilation rate is introduced, which controls the spacing between the values in the kernel.

The receptive field of a neuron can be calculated using the following formula:

 $sli+1=sli+(kernelsize−1)∗dilationfactor$

The receptive field of each neuron in the network increases with each layer, based on the kernel size and dilation rate

## Depthwise Seperable Convolution

Depthwise separable convolution is a type of convolutional operation commonly used to reduce the computational complexity of traditional convolutions while maintaining or even improving the network's performance. It breaks down the convolution into two separate stages: a depthwise convolution and a pointwise convolution

![depthwise](https://github.com/prarthanats/ERA/assets/32382676/d3b3beed-0bf1-46e6-ae6f-9be42ef0839c)

### Depthwise Convolution:

In the depthwise convolution stage, each input channel is convolved with a separate kernel. However, instead of using a single kernel per output channel, depthwise convolution uses a single kernel for each input channel. The depthwise convolution operation applies a small kernel to each input channel independently without mixing information between channels. It performs spatial filtering, capturing spatial correlations within each channel.
As a result, the output of the depthwise convolution has the same number of channels as the input, but the spatial dimensions may change depending on the configuration of the convolutional layer.

### Pointwise Convolution:

After the depthwise convolution, the output is passed through a pointwise convolution. Pointwise convolution is a 1x1 convolution, which means it uses 1x1 kernels.
In the pointwise convolution stage, the purpose is to combine the spatial information from the depthwise convolution output and create new features by applying a set of 1x1 kernels. The number of output channels in the pointwise convolution can be adjusted according to the desired complexity of the network or the number of filters needed.

## Code Structure

The final code can be found here [final code](https://github.com/prarthanats/ERA/blob/main/s9_Assignment/Assignment_9_DPS_DC_Final.ipynb)
The details for this can be found here
1. The code is modularized and has seperate functions
2. Data Augmentation is performed using  Albumentations library. Three techniques are applied in the training data loader: horizontal flipping, shiftScaleRotate, and coarseDropout.[Augmentation](https://github.com/prarthanats/ERA/blob/main/s9_Assignment/data_augmentation.py)
3. Data Loader function downloads, transform the data [Data Loader](https://github.com/prarthanats/ERA/blob/main/s9_Assignment/data_loader.py)
4. model.py file includes Netfunction that is the model structure. It includes a training function and testing function [Model](https://github.com/prarthanats/ERA/blob/main/s9_Assignment/model.py)
4. Visualize.py have a function to plot the metrics, print missclassified images visulaize [Visualize](https://github.com/prarthanats/ERA/blob/main/s9_Assignment/visualize.py)
5. Helper function is used for model summary [Helper](https://github.com/prarthanats/ERA/blob/main/s9_Assignment/helper.py)

## Model Architecture

1. Tried to implement an archiecture similar to GoogLeNet Inception model, with Dilation and Depthwise and added a skip connection to it
2. Input Block - convblock1: Applies a 3x3 convolution with a stride of 1
3. Convolution Block 1 and Convolution Block 2 consists of - 2 convolution layers with **dilation of 2** with padding and 1 convolution layer with stride of 2. Both the blocks have an input channel of 16 and ends with an output channel of 64. 
4. Both the convolution blocks are concatinated using $y = torch.cat((x1, x2), 1)$, the dimension specified is 1, which means the tensors will be concatenated along their columns.
5. Convolution Block 3 is a **Depthwise Seperable Convolution** Block takes an input and output channel of 128 with a 3x3 kernel. It is followed by a **pointwise** convolution with a kernel of 1*1 
6. Convolution Block 4 is a convolution block with 3 convolution for reduction. Here a **skip connection** is added to improve the accuracy, 1 stridded convolution
7. Output Block - **gap** is applied with a kernel size of 5 and a **linear transformation (fully connected layer)** to the output of the average pooling layer to get the target classes

```
       #Input Block, input = 32, Output = 32, RF = 3
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (3, 3), stride = 1, padding = 1, dilation = 1, bias = False),    nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )
        
        #Covolution Block1 , input = 30, Output = 13, RF = 11, Output Channels = 64
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3, 3), stride = 1, padding = 0, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), stride = 1, padding = 0, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3, 3), stride = 2, padding = 1, dilation = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) 
        
        #Covolution Block2 , input = 30, Output = 13, RF = 11, Output Channels = 64
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3, 3), stride = 1, padding = 0, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), stride = 1, padding = 0, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3, 3), stride = 2, padding = 1, dilation = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        )
        
        
        #Covolution Block3 , input = 13, Output = 13, RF = 15, Input Channels = 128, with 64 from CB1 and 64 from CB2 concatenated
        
        self.dsb = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3, 3), padding = 1, groups = 128, bias = False),
            nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = (1, 1), padding = 0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        )
        
        #Covolution Block4 , input = 13, Output = 6, RF = 31
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=(3, 3), stride = 1, padding = 1, dilation = 1, bias = False),
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

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=(3, 3), stride = 2, padding = 0, dilation = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        )
        
        #Output Block , input = 6 , Output = 1, RF = 
        
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size = 6) ## Global Average Pooling
        )

        self.linear = nn.Linear(32, 10)

```
### Model Summary
~~~
	----------------------------------------------------------------
	        Layer (type)               Output Shape         Param #
	================================================================
	            Conv2d-1           [-1, 16, 32, 32]             432
	              ReLU-2           [-1, 16, 32, 32]               0
	       BatchNorm2d-3           [-1, 16, 32, 32]              32
	           Dropout-4           [-1, 16, 32, 32]               0
	            Conv2d-5           [-1, 32, 30, 30]           4,608
	              ReLU-6           [-1, 32, 30, 30]               0
	       BatchNorm2d-7           [-1, 32, 30, 30]              64
	           Dropout-8           [-1, 32, 30, 30]               0
	            Conv2d-9           [-1, 64, 28, 28]          18,432
	             ReLU-10           [-1, 64, 28, 28]               0
	      BatchNorm2d-11           [-1, 64, 28, 28]             128
	          Dropout-12           [-1, 64, 28, 28]               0
	           Conv2d-13           [-1, 64, 13, 13]          36,864
	             ReLU-14           [-1, 64, 13, 13]               0
	      BatchNorm2d-15           [-1, 64, 13, 13]             128
	          Dropout-16           [-1, 64, 13, 13]               0
	           Conv2d-17           [-1, 32, 30, 30]           4,608
	             ReLU-18           [-1, 32, 30, 30]               0
	      BatchNorm2d-19           [-1, 32, 30, 30]              64
	          Dropout-20           [-1, 32, 30, 30]               0
	           Conv2d-21           [-1, 64, 28, 28]          18,432
	             ReLU-22           [-1, 64, 28, 28]               0
	      BatchNorm2d-23           [-1, 64, 28, 28]             128
	          Dropout-24           [-1, 64, 28, 28]               0
	           Conv2d-25           [-1, 64, 13, 13]          36,864
	             ReLU-26           [-1, 64, 13, 13]               0
	      BatchNorm2d-27           [-1, 64, 13, 13]             128
	          Dropout-28           [-1, 64, 13, 13]               0
	           Conv2d-29          [-1, 128, 13, 13]           1,152
	           Conv2d-30           [-1, 32, 13, 13]           4,096
	             ReLU-31           [-1, 32, 13, 13]               0
	      BatchNorm2d-32           [-1, 32, 13, 13]              64
	          Dropout-33           [-1, 32, 13, 13]               0
	           Conv2d-34           [-1, 32, 13, 13]           9,216
	             ReLU-35           [-1, 32, 13, 13]               0
	      BatchNorm2d-36           [-1, 32, 13, 13]              64
	          Dropout-37           [-1, 32, 13, 13]               0
	           Conv2d-38           [-1, 32, 13, 13]           9,216
	             ReLU-39           [-1, 32, 13, 13]               0
	      BatchNorm2d-40           [-1, 32, 13, 13]              64
	          Dropout-41           [-1, 32, 13, 13]               0
	           Conv2d-42             [-1, 32, 6, 6]           9,216
	             ReLU-43             [-1, 32, 6, 6]               0
	      BatchNorm2d-44             [-1, 32, 6, 6]              64
	          Dropout-45             [-1, 32, 6, 6]               0
	        AvgPool2d-46             [-1, 32, 1, 1]               0
	           Linear-47                   [-1, 10]             330
	================================================================
	Total params: 154,394
	Trainable params: 154,394
	Non-trainable params: 0
	----------------------------------------------------------------
	Input size (MB): 0.01
	Forward/backward pass size (MB): 6.68
	Params size (MB): 0.59
	Estimated Total Size (MB): 7.28
	----------------------------------------------------------------
~~~

### Model Graph

![download](https://github.com/prarthanats/ERA/assets/32382676/0f5107c6-d3b9-4b71-a621-6b606a69aafe)

##  Inference

### Implementation
~~~
	Total parameters: 154,394
	Total epochs: 100
	Max Training Accuracy: 88.63, 85% at 66 epoch
	Max Testing Accuracy: 84.20, 85% at 66 epoch
~~~

### Training Logs
~~~		
		
		EPOCH: 1
		Loss=1.5531632900238037 Batch_id=390 Accuracy=39.20: 100%|██████████| 391/391 [00:22<00:00, 17.14it/s]
		
		Test set: Average loss: 0.0103, Accuracy: 5331/10000 (53.31%)
		
		EPOCH: 2
		Loss=1.28281569480896 Batch_id=390 Accuracy=51.42: 100%|██████████| 391/391 [00:22<00:00, 17.54it/s]
		
		Test set: Average loss: 0.0086, Accuracy: 6054/10000 (60.54%)
		
		EPOCH: 3
		Loss=1.2129738330841064 Batch_id=390 Accuracy=56.01: 100%|██████████| 391/391 [00:22<00:00, 17.52it/s]
		
		Test set: Average loss: 0.0078, Accuracy: 6528/10000 (65.28%)
		
		EPOCH: 4
		Loss=1.3539377450942993 Batch_id=390 Accuracy=58.99: 100%|██████████| 391/391 [00:22<00:00, 17.11it/s]
		
		Test set: Average loss: 0.0076, Accuracy: 6590/10000 (65.90%)
		
		EPOCH: 5
		Loss=1.0384457111358643 Batch_id=390 Accuracy=60.66: 100%|██████████| 391/391 [00:20<00:00, 18.77it/s]
		
		Test set: Average loss: 0.0071, Accuracy: 6834/10000 (68.34%)
		
		EPOCH: 6
		Loss=1.0113567113876343 Batch_id=390 Accuracy=62.62: 100%|██████████| 391/391 [00:20<00:00, 19.06it/s]
		
		Test set: Average loss: 0.0068, Accuracy: 6888/10000 (68.88%)
		
		EPOCH: 7
		Loss=1.0294318199157715 Batch_id=390 Accuracy=63.51: 100%|██████████| 391/391 [00:21<00:00, 18.03it/s]
		
		Test set: Average loss: 0.0062, Accuracy: 7299/10000 (72.99%)
		
		EPOCH: 8
		Loss=1.1319397687911987 Batch_id=390 Accuracy=64.51: 100%|██████████| 391/391 [00:21<00:00, 17.87it/s]
		
		Test set: Average loss: 0.0063, Accuracy: 7288/10000 (72.88%)
		
		EPOCH: 9
		Loss=1.131197452545166 Batch_id=390 Accuracy=65.09: 100%|██████████| 391/391 [00:21<00:00, 18.51it/s]
		
		Test set: Average loss: 0.0062, Accuracy: 7333/10000 (73.33%)
		
		EPOCH: 10
		Loss=1.129042387008667 Batch_id=390 Accuracy=65.66: 100%|██████████| 391/391 [00:21<00:00, 18.27it/s]
		
		Test set: Average loss: 0.0064, Accuracy: 7226/10000 (72.26%)
		
		EPOCH: 11
		Loss=1.0685608386993408 Batch_id=390 Accuracy=66.81: 100%|██████████| 391/391 [00:21<00:00, 18.10it/s]
		
		Test set: Average loss: 0.0058, Accuracy: 7506/10000 (75.06%)
		
		EPOCH: 12
		Loss=0.9197486639022827 Batch_id=390 Accuracy=66.98: 100%|██████████| 391/391 [00:22<00:00, 17.73it/s]
		
		Test set: Average loss: 0.0064, Accuracy: 7270/10000 (72.70%)
		
		EPOCH: 13
		Loss=1.1471501588821411 Batch_id=390 Accuracy=67.54: 100%|██████████| 391/391 [00:22<00:00, 17.71it/s]
		
		Test set: Average loss: 0.0056, Accuracy: 7619/10000 (76.19%)
		
		EPOCH: 14
		Loss=0.9363238215446472 Batch_id=390 Accuracy=68.08: 100%|██████████| 391/391 [00:21<00:00, 18.34it/s]
		
		Test set: Average loss: 0.0055, Accuracy: 7642/10000 (76.42%)
		
		EPOCH: 15
		Loss=0.9079523086547852 Batch_id=390 Accuracy=68.40: 100%|██████████| 391/391 [00:20<00:00, 18.69it/s]
		
		Test set: Average loss: 0.0059, Accuracy: 7498/10000 (74.98%)
		
		EPOCH: 16
		Loss=0.9652242660522461 Batch_id=390 Accuracy=69.01: 100%|██████████| 391/391 [00:21<00:00, 18.36it/s]
		
		Test set: Average loss: 0.0053, Accuracy: 7713/10000 (77.13%)
		
		EPOCH: 17
		Loss=0.760948896408081 Batch_id=390 Accuracy=69.19: 100%|██████████| 391/391 [00:22<00:00, 17.57it/s]
		
		Test set: Average loss: 0.0055, Accuracy: 7655/10000 (76.55%)
		
		EPOCH: 18
		Loss=0.9099551439285278 Batch_id=390 Accuracy=69.33: 100%|██████████| 391/391 [00:22<00:00, 17.77it/s]
		
		Test set: Average loss: 0.0054, Accuracy: 7636/10000 (76.36%)
		
		EPOCH: 19
		Loss=1.0128815174102783 Batch_id=390 Accuracy=69.49: 100%|██████████| 391/391 [00:20<00:00, 18.79it/s]
		
		Test set: Average loss: 0.0050, Accuracy: 7821/10000 (78.21%)
		
		EPOCH: 20
		Loss=0.7802087068557739 Batch_id=390 Accuracy=70.08: 100%|██████████| 391/391 [00:20<00:00, 18.67it/s]
		
		Test set: Average loss: 0.0055, Accuracy: 7646/10000 (76.46%)
		
		EPOCH: 21
		Loss=0.962164044380188 Batch_id=390 Accuracy=69.73: 100%|██████████| 391/391 [00:23<00:00, 16.33it/s]
		
		Test set: Average loss: 0.0050, Accuracy: 7767/10000 (77.67%)
		
		EPOCH: 22
		Loss=0.9017642140388489 Batch_id=390 Accuracy=70.35: 100%|██████████| 391/391 [00:22<00:00, 17.70it/s]
		
		Test set: Average loss: 0.0048, Accuracy: 7907/10000 (79.07%)
		
		EPOCH: 23
		Loss=0.9559007883071899 Batch_id=390 Accuracy=70.76: 100%|██████████| 391/391 [00:21<00:00, 17.84it/s]
		
		Test set: Average loss: 0.0048, Accuracy: 7942/10000 (79.42%)
		
		EPOCH: 24
		Loss=0.9057329297065735 Batch_id=390 Accuracy=70.50: 100%|██████████| 391/391 [00:20<00:00, 19.00it/s]
		
		Test set: Average loss: 0.0049, Accuracy: 7838/10000 (78.38%)
		
		EPOCH: 25
		Loss=0.7364867925643921 Batch_id=390 Accuracy=70.84: 100%|██████████| 391/391 [00:21<00:00, 18.62it/s]
		
		Test set: Average loss: 0.0052, Accuracy: 7770/10000 (77.70%)
		
		EPOCH: 26
		Loss=1.0342663526535034 Batch_id=390 Accuracy=70.56: 100%|██████████| 391/391 [00:22<00:00, 17.70it/s]
		
		Test set: Average loss: 0.0049, Accuracy: 7918/10000 (79.18%)
		
		EPOCH: 27
		Loss=0.835205078125 Batch_id=390 Accuracy=71.01: 100%|██████████| 391/391 [00:22<00:00, 17.77it/s]
		
		Test set: Average loss: 0.0052, Accuracy: 7805/10000 (78.05%)
		
		EPOCH: 28
		Loss=1.0152324438095093 Batch_id=390 Accuracy=70.96: 100%|██████████| 391/391 [00:20<00:00, 18.68it/s]
		
		Test set: Average loss: 0.0051, Accuracy: 7799/10000 (77.99%)
		
		EPOCH: 29
		Loss=0.8969658017158508 Batch_id=390 Accuracy=71.21: 100%|██████████| 391/391 [00:20<00:00, 18.81it/s]
		
		Test set: Average loss: 0.0051, Accuracy: 7850/10000 (78.50%)
		
		EPOCH: 30
		Loss=0.7825326919555664 Batch_id=390 Accuracy=71.49: 100%|██████████| 391/391 [00:21<00:00, 18.34it/s]
		
		Test set: Average loss: 0.0048, Accuracy: 7965/10000 (79.65%)
		
		EPOCH: 31
		Loss=0.7856225967407227 Batch_id=390 Accuracy=71.58: 100%|██████████| 391/391 [00:22<00:00, 17.63it/s]
		
		Test set: Average loss: 0.0049, Accuracy: 7848/10000 (78.48%)
		
		EPOCH: 32
		Loss=0.6135531663894653 Batch_id=390 Accuracy=71.37: 100%|██████████| 391/391 [00:21<00:00, 17.82it/s]
		
		Test set: Average loss: 0.0050, Accuracy: 7858/10000 (78.58%)
		
		EPOCH: 33
		Loss=0.8098810315132141 Batch_id=390 Accuracy=71.78: 100%|██████████| 391/391 [00:20<00:00, 18.77it/s]
		
		Test set: Average loss: 0.0050, Accuracy: 7862/10000 (78.62%)
		
		EPOCH: 34
		Loss=0.7225074172019958 Batch_id=390 Accuracy=71.92: 100%|██████████| 391/391 [00:20<00:00, 18.84it/s]
		
		Test set: Average loss: 0.0047, Accuracy: 7937/10000 (79.37%)
		
		EPOCH: 35
		Loss=0.7639008164405823 Batch_id=390 Accuracy=72.25: 100%|██████████| 391/391 [00:22<00:00, 17.63it/s]
		
		Test set: Average loss: 0.0045, Accuracy: 8073/10000 (80.73%)
		
		EPOCH: 36
		Loss=0.5662859678268433 Batch_id=390 Accuracy=72.20: 100%|██████████| 391/391 [00:22<00:00, 17.72it/s]
		
		Test set: Average loss: 0.0050, Accuracy: 7891/10000 (78.91%)
		
		EPOCH: 37
		Loss=0.722066342830658 Batch_id=390 Accuracy=72.44: 100%|██████████| 391/391 [00:21<00:00, 18.52it/s]
		
		Test set: Average loss: 0.0053, Accuracy: 7831/10000 (78.31%)
		
		EPOCH: 38
		Loss=0.7025895118713379 Batch_id=390 Accuracy=72.52: 100%|██████████| 391/391 [00:21<00:00, 18.61it/s]
		
		Test set: Average loss: 0.0049, Accuracy: 7907/10000 (79.07%)
		
		EPOCH: 39
		Loss=0.6721819639205933 Batch_id=390 Accuracy=72.82: 100%|██████████| 391/391 [00:21<00:00, 18.35it/s]
		
		Test set: Average loss: 0.0047, Accuracy: 8000/10000 (80.00%)
		
		EPOCH: 40
		Loss=0.8150988817214966 Batch_id=390 Accuracy=72.60: 100%|██████████| 391/391 [00:22<00:00, 17.19it/s]
		
		Test set: Average loss: 0.0045, Accuracy: 8043/10000 (80.43%)
		
		EPOCH: 41
		Loss=0.7286441922187805 Batch_id=390 Accuracy=73.02: 100%|██████████| 391/391 [00:22<00:00, 17.22it/s]
		
		Test set: Average loss: 0.0046, Accuracy: 8029/10000 (80.29%)
		
		EPOCH: 42
		Loss=0.7282305955886841 Batch_id=390 Accuracy=72.95: 100%|██████████| 391/391 [00:21<00:00, 18.61it/s]
		
		Test set: Average loss: 0.0045, Accuracy: 8081/10000 (80.81%)
		
		EPOCH: 43
		Loss=0.573614239692688 Batch_id=390 Accuracy=73.17: 100%|██████████| 391/391 [00:20<00:00, 18.70it/s]
		
		Test set: Average loss: 0.0047, Accuracy: 8041/10000 (80.41%)
		
		EPOCH: 44
		Loss=0.6928421854972839 Batch_id=390 Accuracy=73.29: 100%|██████████| 391/391 [00:21<00:00, 18.06it/s]
		
		Test set: Average loss: 0.0044, Accuracy: 8119/10000 (81.19%)
		
		EPOCH: 45
		Loss=0.8023465871810913 Batch_id=390 Accuracy=73.46: 100%|██████████| 391/391 [00:22<00:00, 17.42it/s]
		
		Test set: Average loss: 0.0043, Accuracy: 8178/10000 (81.78%)
		
		EPOCH: 46
		Loss=0.7543785572052002 Batch_id=390 Accuracy=73.70: 100%|██████████| 391/391 [00:22<00:00, 17.59it/s]
		
		Test set: Average loss: 0.0044, Accuracy: 8129/10000 (81.29%)
		
		EPOCH: 47
		Loss=0.6195002794265747 Batch_id=390 Accuracy=73.93: 100%|██████████| 391/391 [00:21<00:00, 18.39it/s]
		
		Test set: Average loss: 0.0040, Accuracy: 8272/10000 (82.72%)
		
		EPOCH: 48
		Loss=0.7991483211517334 Batch_id=390 Accuracy=73.80: 100%|██████████| 391/391 [00:21<00:00, 18.42it/s]
		
		Test set: Average loss: 0.0043, Accuracy: 8147/10000 (81.47%)
		
		EPOCH: 49
		Loss=0.760410726070404 Batch_id=390 Accuracy=74.21: 100%|██████████| 391/391 [00:21<00:00, 17.84it/s]
		
		Test set: Average loss: 0.0041, Accuracy: 8226/10000 (82.26%)
		
		EPOCH: 50
		Loss=0.8127638101577759 Batch_id=390 Accuracy=74.66: 100%|██████████| 391/391 [00:21<00:00, 17.79it/s]
		
		Test set: Average loss: 0.0046, Accuracy: 8076/10000 (80.76%)
		
		EPOCH: 51
		Loss=0.8777400255203247 Batch_id=390 Accuracy=74.65: 100%|██████████| 391/391 [00:21<00:00, 18.21it/s]
		
		Test set: Average loss: 0.0042, Accuracy: 8214/10000 (82.14%)
		
		EPOCH: 52
		Loss=0.6903843879699707 Batch_id=390 Accuracy=74.51: 100%|██████████| 391/391 [00:21<00:00, 18.60it/s]
		
		Test set: Average loss: 0.0041, Accuracy: 8248/10000 (82.48%)
		
		EPOCH: 53
		Loss=0.7325760126113892 Batch_id=390 Accuracy=74.96: 100%|██████████| 391/391 [00:21<00:00, 18.20it/s]
		
		Test set: Average loss: 0.0040, Accuracy: 8213/10000 (82.13%)
		
		EPOCH: 54
		Loss=0.7171652317047119 Batch_id=390 Accuracy=75.18: 100%|██████████| 391/391 [00:22<00:00, 17.67it/s]
		
		Test set: Average loss: 0.0043, Accuracy: 8181/10000 (81.81%)
		
		EPOCH: 55
		Loss=0.8022700548171997 Batch_id=390 Accuracy=75.17: 100%|██████████| 391/391 [00:21<00:00, 18.19it/s]
		
		Test set: Average loss: 0.0040, Accuracy: 8297/10000 (82.97%)
		
		EPOCH: 56
		Loss=0.60450679063797 Batch_id=390 Accuracy=75.36: 100%|██████████| 391/391 [00:20<00:00, 19.01it/s]
		
		Test set: Average loss: 0.0040, Accuracy: 8316/10000 (83.16%)
		
		EPOCH: 57
		Loss=0.6327340602874756 Batch_id=390 Accuracy=75.83: 100%|██████████| 391/391 [00:21<00:00, 18.45it/s]
		
		Test set: Average loss: 0.0040, Accuracy: 8303/10000 (83.03%)
		
		EPOCH: 58
		Loss=0.6079310774803162 Batch_id=390 Accuracy=75.79: 100%|██████████| 391/391 [00:22<00:00, 17.39it/s]
		
		Test set: Average loss: 0.0037, Accuracy: 8432/10000 (84.32%)
		
		EPOCH: 59
		Loss=0.668922483921051 Batch_id=390 Accuracy=76.23: 100%|██████████| 391/391 [00:22<00:00, 17.47it/s]
		
		Test set: Average loss: 0.0038, Accuracy: 8409/10000 (84.09%)
		
		EPOCH: 60
		Loss=0.671620786190033 Batch_id=390 Accuracy=76.41: 100%|██████████| 391/391 [00:20<00:00, 18.67it/s]
		
		Test set: Average loss: 0.0038, Accuracy: 8409/10000 (84.09%)
		
		EPOCH: 61
		Loss=0.7455915808677673 Batch_id=390 Accuracy=76.16: 100%|██████████| 391/391 [00:21<00:00, 18.34it/s]
		
		Test set: Average loss: 0.0037, Accuracy: 8411/10000 (84.11%)
		
		EPOCH: 62
		Loss=0.625038206577301 Batch_id=390 Accuracy=76.65: 100%|██████████| 391/391 [00:22<00:00, 17.54it/s]
		
		Test set: Average loss: 0.0039, Accuracy: 8350/10000 (83.50%)
		
		EPOCH: 63
		Loss=0.642417311668396 Batch_id=390 Accuracy=76.87: 100%|██████████| 391/391 [00:22<00:00, 17.55it/s]
		
		Test set: Average loss: 0.0038, Accuracy: 8370/10000 (83.70%)
		
		EPOCH: 64
		Loss=0.7290388345718384 Batch_id=390 Accuracy=77.15: 100%|██████████| 391/391 [00:21<00:00, 18.20it/s]
		
		Test set: Average loss: 0.0036, Accuracy: 8444/10000 (84.44%)
		
		EPOCH: 65
		Loss=0.57587730884552 Batch_id=390 Accuracy=77.64: 100%|██████████| 391/391 [00:21<00:00, 18.54it/s]
		
		Test set: Average loss: 0.0036, Accuracy: 8421/10000 (84.21%)
		
		EPOCH: 66
		Loss=0.546001672744751 Batch_id=390 Accuracy=77.66: 100%|██████████| 391/391 [00:21<00:00, 18.57it/s]
		
		Test set: Average loss: 0.0035, Accuracy: 8514/10000 (85.14%)
		
		EPOCH: 67
		Loss=0.7860480546951294 Batch_id=390 Accuracy=77.88: 100%|██████████| 391/391 [00:22<00:00, 17.67it/s]
		
		Test set: Average loss: 0.0037, Accuracy: 8428/10000 (84.28%)
		
		EPOCH: 68
		Loss=0.8304595947265625 Batch_id=390 Accuracy=78.00: 100%|██████████| 391/391 [00:22<00:00, 17.52it/s]
		
		Test set: Average loss: 0.0035, Accuracy: 8556/10000 (85.56%)
		
		EPOCH: 69
		Loss=0.41816091537475586 Batch_id=390 Accuracy=78.61: 100%|██████████| 391/391 [00:21<00:00, 18.55it/s]
		
		Test set: Average loss: 0.0035, Accuracy: 8531/10000 (85.31%)
		
		EPOCH: 70
		Loss=0.37579289078712463 Batch_id=390 Accuracy=78.63: 100%|██████████| 391/391 [00:21<00:00, 18.56it/s]
		
		Test set: Average loss: 0.0035, Accuracy: 8505/10000 (85.05%)
		
		EPOCH: 71
		Loss=0.6074438095092773 Batch_id=390 Accuracy=78.71: 100%|██████████| 391/391 [00:21<00:00, 17.84it/s]
		
		Test set: Average loss: 0.0034, Accuracy: 8560/10000 (85.60%)
		
		EPOCH: 72
		Loss=0.8565513491630554 Batch_id=390 Accuracy=79.27: 100%|██████████| 391/391 [00:22<00:00, 17.37it/s]
		
		Test set: Average loss: 0.0033, Accuracy: 8597/10000 (85.97%)
		
		EPOCH: 73
		Loss=0.5275697708129883 Batch_id=390 Accuracy=79.55: 100%|██████████| 391/391 [00:22<00:00, 17.49it/s]
		
		Test set: Average loss: 0.0032, Accuracy: 8626/10000 (86.26%)
		
		EPOCH: 74
		Loss=0.5380673408508301 Batch_id=390 Accuracy=79.94: 100%|██████████| 391/391 [00:21<00:00, 18.39it/s]
		
		Test set: Average loss: 0.0031, Accuracy: 8674/10000 (86.74%)
		
		EPOCH: 75
		Loss=0.7614117860794067 Batch_id=390 Accuracy=79.96: 100%|██████████| 391/391 [00:21<00:00, 17.90it/s]
		
		Test set: Average loss: 0.0031, Accuracy: 8661/10000 (86.61%)
		
		EPOCH: 76
		Loss=0.747134804725647 Batch_id=390 Accuracy=80.12: 100%|██████████| 391/391 [00:21<00:00, 18.50it/s]
		
		Test set: Average loss: 0.0032, Accuracy: 8641/10000 (86.41%)
		
		EPOCH: 77
		Loss=0.3993449807167053 Batch_id=390 Accuracy=80.46: 100%|██████████| 391/391 [00:22<00:00, 17.48it/s]
		
		Test set: Average loss: 0.0031, Accuracy: 8664/10000 (86.64%)
		
		EPOCH: 78
		Loss=0.5702266693115234 Batch_id=390 Accuracy=80.68: 100%|██████████| 391/391 [00:22<00:00, 17.56it/s]
		
		Test set: Average loss: 0.0031, Accuracy: 8659/10000 (86.59%)
		
		EPOCH: 79
		Loss=0.4962864816188812 Batch_id=390 Accuracy=81.05: 100%|██████████| 391/391 [00:21<00:00, 18.18it/s]
		
		Test set: Average loss: 0.0031, Accuracy: 8693/10000 (86.93%)
		
		EPOCH: 80
		Loss=0.8814756274223328 Batch_id=390 Accuracy=81.03: 100%|██████████| 391/391 [00:21<00:00, 18.56it/s]
		
		Test set: Average loss: 0.0029, Accuracy: 8728/10000 (87.28%)
		
		EPOCH: 81
		Loss=0.6894984841346741 Batch_id=390 Accuracy=81.44: 100%|██████████| 391/391 [00:22<00:00, 17.74it/s]
		
		Test set: Average loss: 0.0029, Accuracy: 8748/10000 (87.48%)
		
		EPOCH: 82
		Loss=0.5280944108963013 Batch_id=390 Accuracy=81.73: 100%|██████████| 391/391 [00:22<00:00, 17.18it/s]
		
		Test set: Average loss: 0.0029, Accuracy: 8755/10000 (87.55%)
		
		EPOCH: 83
		Loss=0.5541605353355408 Batch_id=390 Accuracy=82.05: 100%|██████████| 391/391 [00:22<00:00, 17.46it/s]
		
		Test set: Average loss: 0.0029, Accuracy: 8779/10000 (87.79%)
		
		EPOCH: 84
		Loss=0.47058120369911194 Batch_id=390 Accuracy=82.25: 100%|██████████| 391/391 [00:21<00:00, 18.44it/s]
		
		Test set: Average loss: 0.0028, Accuracy: 8785/10000 (87.85%)
		
		EPOCH: 85
		Loss=0.6133415102958679 Batch_id=390 Accuracy=82.68: 100%|██████████| 391/391 [00:21<00:00, 18.50it/s]
		
		Test set: Average loss: 0.0028, Accuracy: 8792/10000 (87.92%)
		
		EPOCH: 86
		Loss=0.5670959949493408 Batch_id=390 Accuracy=82.75: 100%|██████████| 391/391 [00:22<00:00, 17.47it/s]
		
		Test set: Average loss: 0.0028, Accuracy: 8803/10000 (88.03%)
		
		EPOCH: 87
		Loss=0.44387561082839966 Batch_id=390 Accuracy=82.77: 100%|██████████| 391/391 [00:22<00:00, 17.53it/s]
		
		Test set: Average loss: 0.0028, Accuracy: 8787/10000 (87.87%)
		
		EPOCH: 88
		Loss=0.32461172342300415 Batch_id=390 Accuracy=82.99: 100%|██████████| 391/391 [00:21<00:00, 18.38it/s]
		
		Test set: Average loss: 0.0027, Accuracy: 8813/10000 (88.13%)
		
		EPOCH: 89
		Loss=0.3450659513473511 Batch_id=390 Accuracy=83.18: 100%|██████████| 391/391 [00:21<00:00, 18.60it/s]
		
		Test set: Average loss: 0.0028, Accuracy: 8850/10000 (88.50%)
		
		EPOCH: 90
		Loss=0.3073321580886841 Batch_id=390 Accuracy=83.37: 100%|██████████| 391/391 [00:21<00:00, 17.94it/s]
		
		Test set: Average loss: 0.0028, Accuracy: 8798/10000 (87.98%)
		
		EPOCH: 91
		Loss=0.324945330619812 Batch_id=390 Accuracy=83.64: 100%|██████████| 391/391 [00:22<00:00, 17.04it/s]
		
		Test set: Average loss: 0.0027, Accuracy: 8850/10000 (88.50%)
		
		EPOCH: 92
		Loss=0.3912912905216217 Batch_id=390 Accuracy=83.86: 100%|██████████| 391/391 [00:23<00:00, 16.99it/s]
		
		Test set: Average loss: 0.0027, Accuracy: 8830/10000 (88.30%)
		
		EPOCH: 93
		Loss=0.40618476271629333 Batch_id=390 Accuracy=83.80: 100%|██████████| 391/391 [00:22<00:00, 17.64it/s]
		
		Test set: Average loss: 0.0027, Accuracy: 8851/10000 (88.51%)
		
		EPOCH: 94
		Loss=0.49034538865089417 Batch_id=390 Accuracy=83.95: 100%|██████████| 391/391 [00:21<00:00, 17.96it/s]
		
		Test set: Average loss: 0.0027, Accuracy: 8855/10000 (88.55%)
		
		EPOCH: 95
		Loss=0.511857271194458 Batch_id=390 Accuracy=84.04: 100%|██████████| 391/391 [00:21<00:00, 17.91it/s]
		
		Test set: Average loss: 0.0027, Accuracy: 8837/10000 (88.37%)
		
		EPOCH: 96
		Loss=0.34483733773231506 Batch_id=390 Accuracy=84.04: 100%|██████████| 391/391 [00:22<00:00, 17.21it/s]
		
		Test set: Average loss: 0.0027, Accuracy: 8844/10000 (88.44%)
		
		EPOCH: 97
		Loss=0.5942537188529968 Batch_id=390 Accuracy=84.04: 100%|██████████| 391/391 [00:22<00:00, 17.01it/s]
		
		Test set: Average loss: 0.0027, Accuracy: 8863/10000 (88.63%)
		
		EPOCH: 98
		Loss=0.501880943775177 Batch_id=390 Accuracy=84.21: 100%|██████████| 391/391 [00:22<00:00, 17.17it/s]
		
		Test set: Average loss: 0.0027, Accuracy: 8839/10000 (88.39%)
		
		EPOCH: 99
		Loss=0.49477800726890564 Batch_id=390 Accuracy=84.38: 100%|██████████| 391/391 [00:21<00:00, 18.02it/s]
		
		Test set: Average loss: 0.0027, Accuracy: 8849/10000 (88.49%)
		
		EPOCH: 100
		Loss=0.5192592144012451 Batch_id=390 Accuracy=84.20: 100%|██████████| 391/391 [00:21<00:00, 18.21it/s]
		
		Test set: Average loss: 0.0027, Accuracy: 8863/10000 (88.63%)	 

~~~

### Accuracy and Loss Plots

![image](https://github.com/prarthanats/ERA/assets/32382676/9d7d0df9-fccc-4127-a8cd-13ac6f434cde)


### Misclassified Images

![image](https://github.com/prarthanats/ERA/assets/32382676/aace216c-5cc0-4df5-87e0-499c789d7552)


### Class Level Accuracy
~~~
	Accuracy of airplane : 91 %
	Accuracy of automobile : 100 %
	Accuracy of  bird : 80 %
	Accuracy of   cat : 85 %
	Accuracy of  deer : 84 %
	Accuracy of   dog : 92 %
	Accuracy of  frog : 96 %
	Accuracy of horse : 91 %
	Accuracy of  ship : 96 %
	Accuracy of truck : 96 %
~~~

## Reference
~~~
	https://medium.com/@RaghavPrabhu/cnn-architectures-lenet-alexnet-vgg-googlenet-and-resnet-7c81c017b848
	http://slazebni.cs.illinois.edu/spring17/lec01_cnn_architectures.pdf
~~~
