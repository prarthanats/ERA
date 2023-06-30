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
4. model.py file includes Netfunction that is the model structure. It includes a training function and testing function [Model] (https://github.com/prarthanats/ERA/blob/main/s9_Assignment/model.py)
4. Visualize.py have a function to plot the metrics, print missclassified images visulaize [Visualize](https://github.com/prarthanats/ERA/blob/main/s9_Assignment/visualize.py)
5. Helper function is used for model summary [Helper](https://github.com/prarthanats/ERA/blob/main/s9_Assignment/helper.py)

## Model Architecture

1. Input Block - convblock1: Applies a 3x3 convolution with a stride of 2
2. Convolution Block 1 and Convolution Block 2 consists of - 2 convolution layers with dilation of 2 with padding and 1 convolution layer with stride of 1. Both the blocks have an input channel of 16 and ends with an output channel of 64. 
3. Both the convolution blocks are concatinated using $y = torch.cat((x1, x2), 1)$, the dimension specified is 1, which means the tensors will be concatenated along their columns.
4. Convolution Block 3 is a Depthwise Seperable Convolution Block takes an input and output channel of 128 with a 3*3 kernel. It is followed by a pointwise convolution with a kernel of 1*1 
5. Convolution Block 4 is a convolution block with 2 convolution for reduction. Here a skip connection is added to improve the accuracy
6. Output Block - gap is applied with a kernel size of 5 and a linear transformation (fully connected layer) to the output of the average pooling layer to get the target classes

(```)
	#Input Block, input = 32, Output = 16, RF = 3
 	self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (3, 3), stride = 2, padding = 1, dilation = 1, bias = False),    nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value))
        
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
(```)
### Model Summary
(```)
-------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 16, 16]             432
              ReLU-2           [-1, 16, 16, 16]               0
       BatchNorm2d-3           [-1, 16, 16, 16]              32
           Dropout-4           [-1, 16, 16, 16]               0
            Conv2d-5           [-1, 32, 14, 14]           4,608
              ReLU-6           [-1, 32, 14, 14]               0
       BatchNorm2d-7           [-1, 32, 14, 14]              64
           Dropout-8           [-1, 32, 14, 14]               0
            Conv2d-9           [-1, 64, 12, 12]          18,432
             ReLU-10           [-1, 64, 12, 12]               0
      BatchNorm2d-11           [-1, 64, 12, 12]             128
          Dropout-12           [-1, 64, 12, 12]               0
           Conv2d-13           [-1, 64, 12, 12]          36,864
             ReLU-14           [-1, 64, 12, 12]               0
      BatchNorm2d-15           [-1, 64, 12, 12]             128
          Dropout-16           [-1, 64, 12, 12]               0
           Conv2d-17           [-1, 32, 14, 14]           4,608
             ReLU-18           [-1, 32, 14, 14]               0
      BatchNorm2d-19           [-1, 32, 14, 14]              64
          Dropout-20           [-1, 32, 14, 14]               0
           Conv2d-21           [-1, 64, 12, 12]          18,432
             ReLU-22           [-1, 64, 12, 12]               0
      BatchNorm2d-23           [-1, 64, 12, 12]             128
          Dropout-24           [-1, 64, 12, 12]               0
           Conv2d-25           [-1, 64, 12, 12]          36,864
             ReLU-26           [-1, 64, 12, 12]               0
      BatchNorm2d-27           [-1, 64, 12, 12]             128
          Dropout-28           [-1, 64, 12, 12]               0
           Conv2d-29          [-1, 128, 10, 10]           1,152
           Conv2d-30           [-1, 32, 10, 10]           4,096
             ReLU-31           [-1, 32, 10, 10]               0
      BatchNorm2d-32           [-1, 32, 10, 10]              64
          Dropout-33           [-1, 32, 10, 10]               0
           Conv2d-34             [-1, 32, 5, 5]           9,216
             ReLU-35             [-1, 32, 5, 5]               0
      BatchNorm2d-36             [-1, 32, 5, 5]              64
          Dropout-37             [-1, 32, 5, 5]               0
           Conv2d-38             [-1, 32, 5, 5]           9,216
             ReLU-39             [-1, 32, 5, 5]               0
      BatchNorm2d-40             [-1, 32, 5, 5]              64
          Dropout-41             [-1, 32, 5, 5]               0
        AvgPool2d-42             [-1, 32, 1, 1]               0
           Linear-43                   [-1, 10]             330
================================================================
Total params: 145,114
Trainable params: 145,114
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.88
Params size (MB): 0.55
Estimated Total Size (MB): 2.44
---------------------------------------------------------------
(```)
### Receptive Field Calculation


### Model Graph


## Implementation and Inference Details

### Model Metrics

	Epochs : 100
	Number of parameters: 145,114 parameters
	LR Scheduler: OneCycleLR
	Final Receptive Field: 47
	Maximum Training Accuracy : 84.73
	Maximum Testing Accuracy : 86.30 (85.07 at 79 epoch)

### Training Log


### Accuracy and Loss Plots


### Misclassified Images


### Class Level Accuracy
