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

1. Input Block - convblock1: Applies a 3x3 convolution with a stride of 2
2. Convolution Block 1 and Convolution Block 2 consists of - 2 convolution layers with dilation of 2 with padding and 1 convolution layer with stride of 1. Both the blocks have an input channel of 16 and ends with an output channel of 64. 
3. Both the convolution blocks are concatinated using $y = torch.cat((x1, x2), 1)$, the dimension specified is 1, which means the tensors will be concatenated along their columns.
4. Convolution Block 3 is a Depthwise Seperable Convolution Block takes an input and output channel of 128 with a 3x3 kernel. It is followed by a pointwise convolution with a kernel of 1*1 
5. Convolution Block 4 is a convolution block with 2 convolution for reduction. Here a skip connection is added to improve the accuracy
6. Output Block - gap is applied with a kernel size of 5 and a linear transformation (fully connected layer) to the output of the average pooling layer to get the target classes

```
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
```
### Model Summary

~~~		
		----------------------------------------------------------------
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
		----------------------------------------------------------------
	 
~~~

### Receptive Field Calculation

Receptive Field for Block 1

<img width="612" alt="rf2" src="https://github.com/prarthanats/ERA/assets/32382676/b300152d-fe97-4821-965a-6e417663af69">

Receptive Field for Block 2

<img width="612" alt="rf2" src="https://github.com/prarthanats/ERA/assets/32382676/61c8a048-a59d-4eaa-bfb5-d55e4936d1d7">

### Model Graph

![download](https://github.com/prarthanats/ERA/assets/32382676/d00f4a1c-dae5-46ed-8411-3347db6cadf5)

## Implementation and Inference Details

### Model Metrics

	Epochs : 100
	Number of parameters: 145,114 parameters
	LR Scheduler: OneCycleLR
	Final Receptive Field: 47
	Maximum Training Accuracy : 84.73
	Maximum Testing Accuracy : 86.30 (85.07 at 79 epoch)

### Training Log
```
	EPOCH: 1
	Loss=1.4609601497650146 Batch_id=390 Accuracy=39.57: 100%|██████████| 391/391 [00:18<00:00, 21.63it/s]
	
	Test set: Average loss: 0.0104, Accuracy: 5274/10000 (52.74%)
	
	EPOCH: 2
	Loss=1.3040621280670166 Batch_id=390 Accuracy=51.67: 100%|██████████| 391/391 [00:19<00:00, 19.74it/s]
	
	Test set: Average loss: 0.0089, Accuracy: 5957/10000 (59.57%)
	
	EPOCH: 3
	Loss=1.387711763381958 Batch_id=390 Accuracy=55.58: 100%|██████████| 391/391 [00:18<00:00, 21.19it/s]
	
	Test set: Average loss: 0.0081, Accuracy: 6402/10000 (64.02%)
	
	EPOCH: 4
	Loss=0.9404183626174927 Batch_id=390 Accuracy=58.01: 100%|██████████| 391/391 [00:19<00:00, 19.89it/s]
	
	Test set: Average loss: 0.0078, Accuracy: 6557/10000 (65.57%)
	
	EPOCH: 5
	Loss=1.3025163412094116 Batch_id=390 Accuracy=59.79: 100%|██████████| 391/391 [00:17<00:00, 22.15it/s]
	
	Test set: Average loss: 0.0073, Accuracy: 6724/10000 (67.24%)
	
	EPOCH: 6
	Loss=1.2056061029434204 Batch_id=390 Accuracy=61.46: 100%|██████████| 391/391 [00:18<00:00, 20.73it/s]
	
	Test set: Average loss: 0.0070, Accuracy: 6863/10000 (68.63%)
	
	EPOCH: 7
	Loss=1.1469991207122803 Batch_id=390 Accuracy=62.92: 100%|██████████| 391/391 [00:17<00:00, 22.32it/s]
	
	Test set: Average loss: 0.0069, Accuracy: 6924/10000 (69.24%)
	
	EPOCH: 8
	Loss=1.053687572479248 Batch_id=390 Accuracy=63.59: 100%|██████████| 391/391 [00:18<00:00, 21.62it/s]
	
	Test set: Average loss: 0.0070, Accuracy: 6896/10000 (68.96%)
	
	EPOCH: 9
	Loss=1.1110724210739136 Batch_id=390 Accuracy=64.46: 100%|██████████| 391/391 [00:17<00:00, 22.56it/s]
	
	Test set: Average loss: 0.0065, Accuracy: 7150/10000 (71.50%)
	
	EPOCH: 10
	Loss=0.9890600442886353 Batch_id=390 Accuracy=65.37: 100%|██████████| 391/391 [00:18<00:00, 21.26it/s]
	
	Test set: Average loss: 0.0064, Accuracy: 7222/10000 (72.22%)
	
	EPOCH: 11
	Loss=0.9511944055557251 Batch_id=390 Accuracy=66.14: 100%|██████████| 391/391 [00:17<00:00, 22.52it/s]
	
	Test set: Average loss: 0.0063, Accuracy: 7197/10000 (71.97%)
	
	EPOCH: 12
	Loss=0.788652241230011 Batch_id=390 Accuracy=66.47: 100%|██████████| 391/391 [00:18<00:00, 20.82it/s]
	
	Test set: Average loss: 0.0061, Accuracy: 7298/10000 (72.98%)
	
	EPOCH: 13
	Loss=1.044738531112671 Batch_id=390 Accuracy=66.91: 100%|██████████| 391/391 [00:17<00:00, 22.04it/s]
	
	Test set: Average loss: 0.0058, Accuracy: 7473/10000 (74.73%)
	
	EPOCH: 14
	Loss=0.9132393002510071 Batch_id=390 Accuracy=67.14: 100%|██████████| 391/391 [00:18<00:00, 21.29it/s]
	
	Test set: Average loss: 0.0062, Accuracy: 7344/10000 (73.44%)
	
	EPOCH: 15
	Loss=0.9120794534683228 Batch_id=390 Accuracy=67.76: 100%|██████████| 391/391 [00:17<00:00, 22.24it/s]
	
	Test set: Average loss: 0.0056, Accuracy: 7523/10000 (75.23%)
	
	EPOCH: 16
	Loss=1.058044195175171 Batch_id=390 Accuracy=68.34: 100%|██████████| 391/391 [00:18<00:00, 20.67it/s]
	
	Test set: Average loss: 0.0056, Accuracy: 7578/10000 (75.78%)
	
	EPOCH: 17
	Loss=1.1241978406906128 Batch_id=390 Accuracy=68.40: 100%|██████████| 391/391 [00:17<00:00, 22.14it/s]
	
	Test set: Average loss: 0.0056, Accuracy: 7580/10000 (75.80%)
	
	EPOCH: 18
	Loss=0.8938026428222656 Batch_id=390 Accuracy=69.16: 100%|██████████| 391/391 [00:18<00:00, 20.87it/s]
	
	Test set: Average loss: 0.0055, Accuracy: 7613/10000 (76.13%)
	
	EPOCH: 19
	Loss=0.927864670753479 Batch_id=390 Accuracy=69.54: 100%|██████████| 391/391 [00:17<00:00, 22.23it/s]
	
	Test set: Average loss: 0.0057, Accuracy: 7514/10000 (75.14%)
	
	EPOCH: 20
	Loss=1.2750142812728882 Batch_id=390 Accuracy=69.55: 100%|██████████| 391/391 [00:19<00:00, 20.06it/s]
	
	Test set: Average loss: 0.0057, Accuracy: 7557/10000 (75.57%)
	
	EPOCH: 21
	Loss=0.9408542513847351 Batch_id=390 Accuracy=69.42: 100%|██████████| 391/391 [00:17<00:00, 21.76it/s]
	
	Test set: Average loss: 0.0057, Accuracy: 7502/10000 (75.02%)
	
	EPOCH: 22
	Loss=0.9009097218513489 Batch_id=390 Accuracy=69.70: 100%|██████████| 391/391 [00:19<00:00, 20.51it/s]
	
	Test set: Average loss: 0.0053, Accuracy: 7640/10000 (76.40%)
	
	EPOCH: 23
	Loss=0.7435827851295471 Batch_id=390 Accuracy=70.10: 100%|██████████| 391/391 [00:17<00:00, 22.19it/s]
	
	Test set: Average loss: 0.0054, Accuracy: 7624/10000 (76.24%)
	
	EPOCH: 24
	Loss=0.7392395734786987 Batch_id=390 Accuracy=70.09: 100%|██████████| 391/391 [00:18<00:00, 20.66it/s]
	
	Test set: Average loss: 0.0057, Accuracy: 7503/10000 (75.03%)
	
	EPOCH: 25
	Loss=0.9236645698547363 Batch_id=390 Accuracy=70.29: 100%|██████████| 391/391 [00:17<00:00, 21.91it/s]
	
	Test set: Average loss: 0.0060, Accuracy: 7453/10000 (74.53%)
	
	EPOCH: 26
	Loss=0.7633862495422363 Batch_id=390 Accuracy=70.29: 100%|██████████| 391/391 [00:18<00:00, 20.95it/s]
	
	Test set: Average loss: 0.0052, Accuracy: 7756/10000 (77.56%)
	
	EPOCH: 27
	Loss=0.7709025740623474 Batch_id=390 Accuracy=70.27: 100%|██████████| 391/391 [00:17<00:00, 22.36it/s]
	
	Test set: Average loss: 0.0055, Accuracy: 7653/10000 (76.53%)
	
	EPOCH: 28
	Loss=0.9363770484924316 Batch_id=390 Accuracy=70.58: 100%|██████████| 391/391 [00:18<00:00, 21.09it/s]
	
	Test set: Average loss: 0.0054, Accuracy: 7690/10000 (76.90%)
	
	EPOCH: 29
	Loss=0.6948153972625732 Batch_id=390 Accuracy=70.65: 100%|██████████| 391/391 [00:17<00:00, 22.31it/s]
	
	Test set: Average loss: 0.0054, Accuracy: 7588/10000 (75.88%)
	
	EPOCH: 30
	Loss=1.0303750038146973 Batch_id=390 Accuracy=70.62: 100%|██████████| 391/391 [00:18<00:00, 21.48it/s]
	
	Test set: Average loss: 0.0055, Accuracy: 7631/10000 (76.31%)
	
	EPOCH: 31
	Loss=0.7791464924812317 Batch_id=390 Accuracy=70.94: 100%|██████████| 391/391 [00:17<00:00, 22.47it/s]
	
	Test set: Average loss: 0.0052, Accuracy: 7812/10000 (78.12%)
	
	EPOCH: 32
	Loss=0.7966980934143066 Batch_id=390 Accuracy=71.38: 100%|██████████| 391/391 [00:17<00:00, 22.06it/s]
	
	Test set: Average loss: 0.0052, Accuracy: 7812/10000 (78.12%)
	
	EPOCH: 33
	Loss=0.8082650303840637 Batch_id=390 Accuracy=71.47: 100%|██████████| 391/391 [00:17<00:00, 22.21it/s]
	
	Test set: Average loss: 0.0053, Accuracy: 7683/10000 (76.83%)
	
	EPOCH: 34
	Loss=0.7246282696723938 Batch_id=390 Accuracy=71.12: 100%|██████████| 391/391 [00:17<00:00, 22.25it/s]
	
	Test set: Average loss: 0.0050, Accuracy: 7868/10000 (78.68%)
	
	EPOCH: 35
	Loss=1.084949016571045 Batch_id=390 Accuracy=71.56: 100%|██████████| 391/391 [00:17<00:00, 22.30it/s]
	
	Test set: Average loss: 0.0054, Accuracy: 7745/10000 (77.45%)
	
	EPOCH: 36
	Loss=0.859399139881134 Batch_id=390 Accuracy=71.66: 100%|██████████| 391/391 [00:17<00:00, 22.20it/s]
	
	Test set: Average loss: 0.0055, Accuracy: 7671/10000 (76.71%)
	
	EPOCH: 37
	Loss=0.8666099309921265 Batch_id=390 Accuracy=72.10: 100%|██████████| 391/391 [00:17<00:00, 22.14it/s]
	
	Test set: Average loss: 0.0048, Accuracy: 7874/10000 (78.74%)
	
	EPOCH: 38
	Loss=0.7457512021064758 Batch_id=390 Accuracy=71.99: 100%|██████████| 391/391 [00:17<00:00, 22.10it/s]
	
	Test set: Average loss: 0.0053, Accuracy: 7790/10000 (77.90%)
	
	EPOCH: 39
	Loss=1.0506011247634888 Batch_id=390 Accuracy=72.45: 100%|██████████| 391/391 [00:17<00:00, 22.40it/s]
	
	Test set: Average loss: 0.0049, Accuracy: 7842/10000 (78.42%)
	
	EPOCH: 40
	Loss=0.8835695385932922 Batch_id=390 Accuracy=72.40: 100%|██████████| 391/391 [00:17<00:00, 21.97it/s]
	
	Test set: Average loss: 0.0050, Accuracy: 7856/10000 (78.56%)
	
	EPOCH: 41
	Loss=0.7087905406951904 Batch_id=390 Accuracy=72.40: 100%|██████████| 391/391 [00:18<00:00, 21.18it/s]
	
	Test set: Average loss: 0.0053, Accuracy: 7732/10000 (77.32%)
	
	EPOCH: 42
	Loss=0.8787147402763367 Batch_id=390 Accuracy=72.64: 100%|██████████| 391/391 [00:17<00:00, 22.19it/s]
	
	Test set: Average loss: 0.0049, Accuracy: 7882/10000 (78.82%)
	
	EPOCH: 43
	Loss=1.0232592821121216 Batch_id=390 Accuracy=72.75: 100%|██████████| 391/391 [00:19<00:00, 20.16it/s]
	
	Test set: Average loss: 0.0049, Accuracy: 7914/10000 (79.14%)
	
	EPOCH: 44
	Loss=0.6632324457168579 Batch_id=390 Accuracy=73.13: 100%|██████████| 391/391 [00:17<00:00, 22.06it/s]
	
	Test set: Average loss: 0.0050, Accuracy: 7849/10000 (78.49%)
	
	EPOCH: 45
	Loss=0.7201938629150391 Batch_id=390 Accuracy=73.27: 100%|██████████| 391/391 [00:19<00:00, 20.52it/s]
	
	Test set: Average loss: 0.0048, Accuracy: 7868/10000 (78.68%)
	
	EPOCH: 46
	Loss=0.7488065361976624 Batch_id=390 Accuracy=73.37: 100%|██████████| 391/391 [00:17<00:00, 22.03it/s]
	
	Test set: Average loss: 0.0050, Accuracy: 7780/10000 (77.80%)
	
	EPOCH: 47
	Loss=0.7389547824859619 Batch_id=390 Accuracy=73.81: 100%|██████████| 391/391 [00:19<00:00, 20.37it/s]
	
	Test set: Average loss: 0.0048, Accuracy: 7944/10000 (79.44%)
	
	EPOCH: 48
	Loss=0.9148517847061157 Batch_id=390 Accuracy=73.65: 100%|██████████| 391/391 [00:17<00:00, 22.37it/s]
	
	Test set: Average loss: 0.0046, Accuracy: 8001/10000 (80.01%)
	
	EPOCH: 49
	Loss=0.8983342051506042 Batch_id=390 Accuracy=73.87: 100%|██████████| 391/391 [00:19<00:00, 20.56it/s]
	
	Test set: Average loss: 0.0046, Accuracy: 8021/10000 (80.21%)
	
	EPOCH: 50
	Loss=0.5360952615737915 Batch_id=390 Accuracy=74.35: 100%|██████████| 391/391 [00:17<00:00, 21.92it/s]
	
	Test set: Average loss: 0.0051, Accuracy: 7816/10000 (78.16%)
	
	EPOCH: 51
	Loss=0.6336027383804321 Batch_id=390 Accuracy=74.30: 100%|██████████| 391/391 [00:19<00:00, 20.55it/s]
	
	Test set: Average loss: 0.0046, Accuracy: 7996/10000 (79.96%)
	
	EPOCH: 52
	Loss=0.9709226489067078 Batch_id=390 Accuracy=74.60: 100%|██████████| 391/391 [00:18<00:00, 21.37it/s]
	
	Test set: Average loss: 0.0048, Accuracy: 7923/10000 (79.23%)
	
	EPOCH: 53
	Loss=0.7859131097793579 Batch_id=390 Accuracy=74.82: 100%|██████████| 391/391 [00:19<00:00, 19.86it/s]
	
	Test set: Average loss: 0.0045, Accuracy: 8008/10000 (80.08%)
	
	EPOCH: 54
	Loss=0.8552873730659485 Batch_id=390 Accuracy=75.02: 100%|██████████| 391/391 [00:18<00:00, 21.50it/s]
	
	Test set: Average loss: 0.0044, Accuracy: 8142/10000 (81.42%)
	
	EPOCH: 55
	Loss=0.9360536336898804 Batch_id=390 Accuracy=74.92: 100%|██████████| 391/391 [00:17<00:00, 21.95it/s]
	
	Test set: Average loss: 0.0047, Accuracy: 7971/10000 (79.71%)
	
	EPOCH: 56
	Loss=0.870067298412323 Batch_id=390 Accuracy=74.99: 100%|██████████| 391/391 [00:17<00:00, 22.36it/s]
	
	Test set: Average loss: 0.0045, Accuracy: 8081/10000 (80.81%)
	
	EPOCH: 57
	Loss=0.7957128286361694 Batch_id=390 Accuracy=75.66: 100%|██████████| 391/391 [00:17<00:00, 22.04it/s]
	
	Test set: Average loss: 0.0045, Accuracy: 8050/10000 (80.50%)
	
	EPOCH: 58
	Loss=0.858467698097229 Batch_id=390 Accuracy=75.62: 100%|██████████| 391/391 [00:17<00:00, 22.75it/s]
	
	Test set: Average loss: 0.0044, Accuracy: 8058/10000 (80.58%)
	
	EPOCH: 59
	Loss=0.719828724861145 Batch_id=390 Accuracy=76.21: 100%|██████████| 391/391 [00:17<00:00, 21.76it/s]
	
	Test set: Average loss: 0.0043, Accuracy: 8129/10000 (81.29%)
	
	EPOCH: 60
	Loss=0.6769415140151978 Batch_id=390 Accuracy=75.97: 100%|██████████| 391/391 [00:17<00:00, 22.65it/s]
	
	Test set: Average loss: 0.0043, Accuracy: 8144/10000 (81.44%)
	
	EPOCH: 61
	Loss=0.6273127198219299 Batch_id=390 Accuracy=76.58: 100%|██████████| 391/391 [00:17<00:00, 21.92it/s]
	
	Test set: Average loss: 0.0042, Accuracy: 8180/10000 (81.80%)
	
	EPOCH: 62
	Loss=0.8165104985237122 Batch_id=390 Accuracy=76.92: 100%|██████████| 391/391 [00:17<00:00, 21.98it/s]
	
	Test set: Average loss: 0.0041, Accuracy: 8215/10000 (82.15%)
	
	EPOCH: 63
	Loss=0.6025989055633545 Batch_id=390 Accuracy=77.04: 100%|██████████| 391/391 [00:18<00:00, 21.47it/s]
	
	Test set: Average loss: 0.0041, Accuracy: 8247/10000 (82.47%)
	
	EPOCH: 64
	Loss=0.608680784702301 Batch_id=390 Accuracy=77.43: 100%|██████████| 391/391 [00:17<00:00, 22.58it/s]
	
	Test set: Average loss: 0.0042, Accuracy: 8165/10000 (81.65%)
	
	EPOCH: 65
	Loss=0.5596030950546265 Batch_id=390 Accuracy=77.47: 100%|██████████| 391/391 [00:17<00:00, 21.74it/s]
	
	Test set: Average loss: 0.0039, Accuracy: 8282/10000 (82.82%)
	
	EPOCH: 66
	Loss=0.5544354915618896 Batch_id=390 Accuracy=77.75: 100%|██████████| 391/391 [00:17<00:00, 22.59it/s]
	
	Test set: Average loss: 0.0040, Accuracy: 8281/10000 (82.81%)
	
	EPOCH: 67
	Loss=0.7951173186302185 Batch_id=390 Accuracy=77.78: 100%|██████████| 391/391 [00:17<00:00, 22.03it/s]
	
	Test set: Average loss: 0.0041, Accuracy: 8243/10000 (82.43%)
	
	EPOCH: 68
	Loss=0.6467259526252747 Batch_id=390 Accuracy=78.54: 100%|██████████| 391/391 [00:17<00:00, 22.58it/s]
	
	Test set: Average loss: 0.0041, Accuracy: 8224/10000 (82.24%)
	
	EPOCH: 69
	Loss=0.6803727746009827 Batch_id=390 Accuracy=78.48: 100%|██████████| 391/391 [00:17<00:00, 21.79it/s]
	
	Test set: Average loss: 0.0039, Accuracy: 8308/10000 (83.08%)
	
	EPOCH: 70
	Loss=0.7241876125335693 Batch_id=390 Accuracy=78.92: 100%|██████████| 391/391 [00:17<00:00, 22.53it/s]
	
	Test set: Average loss: 0.0039, Accuracy: 8349/10000 (83.49%)
	
	EPOCH: 71
	Loss=0.4764552116394043 Batch_id=390 Accuracy=79.26: 100%|██████████| 391/391 [00:18<00:00, 21.41it/s]
	
	Test set: Average loss: 0.0038, Accuracy: 8313/10000 (83.13%)
	
	EPOCH: 72
	Loss=0.5246113538742065 Batch_id=390 Accuracy=79.34: 100%|██████████| 391/391 [00:17<00:00, 21.78it/s]
	
	Test set: Average loss: 0.0037, Accuracy: 8394/10000 (83.94%)
	
	EPOCH: 73
	Loss=0.6209293603897095 Batch_id=390 Accuracy=79.72: 100%|██████████| 391/391 [00:17<00:00, 22.16it/s]
	
	Test set: Average loss: 0.0037, Accuracy: 8411/10000 (84.11%)
	
	EPOCH: 74
	Loss=0.5719833970069885 Batch_id=390 Accuracy=79.96: 100%|██████████| 391/391 [00:17<00:00, 22.19it/s]
	
	Test set: Average loss: 0.0037, Accuracy: 8408/10000 (84.08%)
	
	EPOCH: 75
	Loss=0.4492349624633789 Batch_id=390 Accuracy=80.32: 100%|██████████| 391/391 [00:17<00:00, 21.99it/s]
	
	Test set: Average loss: 0.0038, Accuracy: 8394/10000 (83.94%)
	
	EPOCH: 76
	Loss=0.6055085062980652 Batch_id=390 Accuracy=80.48: 100%|██████████| 391/391 [00:17<00:00, 22.31it/s]
	
	Test set: Average loss: 0.0036, Accuracy: 8463/10000 (84.63%)
	
	EPOCH: 77
	Loss=0.546164870262146 Batch_id=390 Accuracy=80.82: 100%|██████████| 391/391 [00:17<00:00, 21.85it/s]
	
	Test set: Average loss: 0.0036, Accuracy: 8460/10000 (84.60%)
	
	EPOCH: 78
	Loss=0.4961410164833069 Batch_id=390 Accuracy=81.20: 100%|██████████| 391/391 [00:17<00:00, 22.17it/s]
	
	Test set: Average loss: 0.0036, Accuracy: 8447/10000 (84.47%)
	
	EPOCH: 79
	Loss=0.6026605367660522 Batch_id=390 Accuracy=81.59: 100%|██████████| 391/391 [00:17<00:00, 22.02it/s]
	
	Test set: Average loss: 0.0035, Accuracy: 8507/10000 (85.07%)
	
	EPOCH: 80
	Loss=0.4251566529273987 Batch_id=390 Accuracy=81.78: 100%|██████████| 391/391 [00:17<00:00, 21.89it/s]
	
	Test set: Average loss: 0.0036, Accuracy: 8530/10000 (85.30%)
	
	EPOCH: 81
	Loss=0.5538046360015869 Batch_id=390 Accuracy=82.08: 100%|██████████| 391/391 [00:17<00:00, 22.11it/s]
	
	Test set: Average loss: 0.0035, Accuracy: 8519/10000 (85.19%)
	
	EPOCH: 82
	Loss=0.46853381395339966 Batch_id=390 Accuracy=82.42: 100%|██████████| 391/391 [00:17<00:00, 21.80it/s]
	
	Test set: Average loss: 0.0034, Accuracy: 8533/10000 (85.33%)
	
	EPOCH: 83
	Loss=0.4316166341304779 Batch_id=390 Accuracy=82.65: 100%|██████████| 391/391 [00:17<00:00, 22.14it/s]
	
	Test set: Average loss: 0.0035, Accuracy: 8509/10000 (85.09%)
	
	EPOCH: 84
	Loss=0.5565477609634399 Batch_id=390 Accuracy=82.66: 100%|██████████| 391/391 [00:18<00:00, 20.78it/s]
	
	Test set: Average loss: 0.0035, Accuracy: 8542/10000 (85.42%)
	
	EPOCH: 85
	Loss=0.5018292665481567 Batch_id=390 Accuracy=82.85: 100%|██████████| 391/391 [00:17<00:00, 22.01it/s]
	
	Test set: Average loss: 0.0034, Accuracy: 8598/10000 (85.98%)
	
	EPOCH: 86
	Loss=0.6028844118118286 Batch_id=390 Accuracy=83.36: 100%|██████████| 391/391 [00:18<00:00, 20.89it/s]
	
	Test set: Average loss: 0.0034, Accuracy: 8586/10000 (85.86%)
	
	EPOCH: 87
	Loss=0.6318896412849426 Batch_id=390 Accuracy=83.38: 100%|██████████| 391/391 [00:17<00:00, 21.97it/s]
	
	Test set: Average loss: 0.0034, Accuracy: 8580/10000 (85.80%)
	
	EPOCH: 88
	Loss=0.4518156945705414 Batch_id=390 Accuracy=83.74: 100%|██████████| 391/391 [00:18<00:00, 21.25it/s]
	
	Test set: Average loss: 0.0033, Accuracy: 8609/10000 (86.09%)
	
	EPOCH: 89
	Loss=0.6676636338233948 Batch_id=390 Accuracy=83.71: 100%|██████████| 391/391 [00:17<00:00, 22.08it/s]
	
	Test set: Average loss: 0.0033, Accuracy: 8610/10000 (86.10%)
	
	EPOCH: 90
	Loss=0.535239577293396 Batch_id=390 Accuracy=84.08: 100%|██████████| 391/391 [00:18<00:00, 21.36it/s]
	
	Test set: Average loss: 0.0033, Accuracy: 8611/10000 (86.11%)
	
	EPOCH: 91
	Loss=0.4955207407474518 Batch_id=390 Accuracy=84.26: 100%|██████████| 391/391 [00:17<00:00, 21.99it/s]
	
	Test set: Average loss: 0.0033, Accuracy: 8607/10000 (86.07%)
	
	EPOCH: 92
	Loss=0.428192675113678 Batch_id=390 Accuracy=83.93: 100%|██████████| 391/391 [00:18<00:00, 20.70it/s]
	
	Test set: Average loss: 0.0033, Accuracy: 8624/10000 (86.24%)
	
	EPOCH: 93
	Loss=0.4620881676673889 Batch_id=390 Accuracy=84.62: 100%|██████████| 391/391 [00:17<00:00, 22.10it/s]
	
	Test set: Average loss: 0.0033, Accuracy: 8613/10000 (86.13%)
	
	EPOCH: 94
	Loss=0.4027479290962219 Batch_id=390 Accuracy=84.58: 100%|██████████| 391/391 [00:18<00:00, 20.89it/s]
	
	Test set: Average loss: 0.0033, Accuracy: 8624/10000 (86.24%)
	
	EPOCH: 95
	Loss=0.5361292362213135 Batch_id=390 Accuracy=84.64: 100%|██████████| 391/391 [00:18<00:00, 21.70it/s]
	
	Test set: Average loss: 0.0033, Accuracy: 8641/10000 (86.41%)
	
	EPOCH: 96
	Loss=0.4031130373477936 Batch_id=390 Accuracy=84.65: 100%|██████████| 391/391 [00:19<00:00, 20.41it/s]
	
	Test set: Average loss: 0.0033, Accuracy: 8628/10000 (86.28%)
	
	EPOCH: 97
	Loss=0.6068166494369507 Batch_id=390 Accuracy=84.83: 100%|██████████| 391/391 [00:17<00:00, 21.97it/s]
	
	Test set: Average loss: 0.0032, Accuracy: 8641/10000 (86.41%)
	
	EPOCH: 98
	Loss=0.3092423379421234 Batch_id=390 Accuracy=84.86: 100%|██████████| 391/391 [00:18<00:00, 20.80it/s]
	
	Test set: Average loss: 0.0033, Accuracy: 8632/10000 (86.32%)
	
	EPOCH: 99
	Loss=0.46150994300842285 Batch_id=390 Accuracy=84.68: 100%|██████████| 391/391 [00:17<00:00, 22.12it/s]
	
	Test set: Average loss: 0.0033, Accuracy: 8631/10000 (86.31%)
	
	EPOCH: 100
	Loss=0.27930739521980286 Batch_id=390 Accuracy=84.73: 100%|██████████| 391/391 [00:18<00:00, 20.62it/s]
	
	Test set: Average loss: 0.0033, Accuracy: 8630/10000 (86.30%)

```

### Accuracy and Loss Plots

![image](https://github.com/prarthanats/ERA/assets/32382676/b6b4b309-25ca-425c-a19d-f7b7b4a42814)

### Misclassified Images

![image](https://github.com/prarthanats/ERA/assets/32382676/64b4cdcd-b70c-4de0-9a46-c126ec225515)

### Class Level Accuracy
'''
	Accuracy of airplane : 80 %
	Accuracy of automobile : 94 %
	Accuracy of  bird : 89 %
	Accuracy of   cat : 54 %
	Accuracy of  deer : 93 %
	Accuracy of   dog : 60 %
	Accuracy of  frog : 91 %
	Accuracy of horse : 92 %
	Accuracy of  ship : 96 %
	Accuracy of truck : 94 %
'''
