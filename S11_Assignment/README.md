![Loss and Accuracy](https://github.com/prarthanats/ERA/assets/32382676/52275c67-6f5d-4561-9632-3ab93d5d0f4a)# Application of CAMs, LRs and Optimizers on CIFAR 10 Dataset 

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


## Model Architecture 

ResNet18 is a convolutional neural network (CNN) architecture that was introduced in the paper "Deep Residual Learning for Image Recognition" by He et al. (2015). It is a relatively shallow network, with only 18 layers, but it is still able to achieve state-of-the-art results on image classification tasks.

The key innovation of ResNet is the use of residual blocks, which allow for the training of very deep networks by addressing the vanishing gradient problem. The residual blocks introduce skip connections that enable the network to learn residual mappings, making it easier to optimize.

It consists of several layers of BasicBlocks, and each BasicBlock contains two 3x3 convolutional layers with batch normalization and a residual connection.

~~~
	1. BasicBlock Class:
		The BasicBlock class represents the building block of the ResNet-18 model.
		It inherits from nn.Module, making it a PyTorch module.
		The expansion attribute is set to 1, indicating that the number of output channels is the same as the number of input channels.
	
		The __init__ method initializes the block and defines its layers:
			self.conv1,self.conv2 : The 3x3 convolutional layer with in_planes input channels and planes output channels.
			self.bn1, self.bn2: Batch normalization after the convolutional layer.
			self.shortcut: The shortcut connection for the residual block. If the stride is not 1 or the number of input channels (in_planes) is not equal to self.expansion*planes, a 1x1 convolution is used to match the dimensions, followed by batch normalization.
		The forward method defines the forward pass of the BasicBlock:
			The input x is passed through the first convolutional layer, batch normalization, and ReLU activation.
			The result is then passed through the second convolutional layer and batch normalization.
			The output from the second convolutional layer is added to the shortcut connection, creating the residual connection.
			The result is passed through the ReLU activation function, and the final output is returned.
	2. ResNet Class:
	
		The ResNet class represents the overall ResNet-18 model.
		It also inherits from nn.Module.
		The __init__ method initializes the ResNet-18 model and defines its layers:
			self.conv1: The initial 3x3 convolutional layer that takes input images with 3 channels (RGB) and produces 64 output channels.
			self.bn1: Batch normalization after the first convolutional layer.
			self.layer1, self.layer2, self.layer3, and self.layer4: Four stages of the ResNet-18 model, each consisting of multiple BasicBlocks.
			self.linear: The fully connected layer at the end that maps the final feature vectors to the number of output classes (num_classes).
		The _make_layer method is a helper function that creates a stage of BasicBlocks with the specified number of blocks, planes (output channels), and stride.
		The forward method defines the forward pass of the ResNet-18 model:
			The input x is passed through the first convolutional layer, batch normalization, and ReLU activation.
			It is then passed through each stage (self.layer1, self.layer2, etc.) one by one.
			After the last stage (self.layer4), the output is passed through an average pooling layer to reduce the spatial dimensions to 1x1.
			The output is then flattened and passed through the fully connected layer to get the final classification logits.

	3. ResNet18 Function:
	
	The ResNet18 function creates an instance of the ResNet class with BasicBlocks and the specific number of blocks in each stage [2, 2, 2, 2].
	It returns the ResNet-18 model, which is ready for training and evaluation.

~~~

The final model can be visualized as:

![Graph](https://github.com/prarthanats/ERA/assets/32382676/fa20cae6-713c-4b12-8469-820858531e44)

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


### Learning Rate Range Test Curve

![LR](https://github.com/prarthanats/ERA/assets/32382676/26e49457-94e0-4e77-b540-db8feedcc7e2)

### Accuracy and Loss Plots

![Loss and Accuracy](https://github.com/prarthanats/ERA/assets/32382676/41761179-2f61-4bf9-85b0-b43d97b7af84)

### Misclassified Images

![Misclassify](https://github.com/prarthanats/ERA/assets/32382676/c3b1d27c-255e-4a4e-89f7-f2c196cfae7d)

### GradCam Misclassified Images

![GradCam](https://github.com/prarthanats/ERA/assets/32382676/5a049982-410b-4167-9219-2ecf203b981a)

### TensorBoard Outputs

<img width="537" alt="tensor Output" src="https://github.com/prarthanats/ERA/assets/32382676/df466f7d-3116-4ce5-b667-2d121693dc75">


### Training Logs
~~~
	Epoch 1:
	Loss=1.4932713508605957 Batch_id=97 LR=0.00351 Accuracy=37.80: 100%|██████████| 98/98 [00:39<00:00,  2.49it/s]
	
	Test set: Average loss: 0.0041, Accuracy: 3605/10000 (36.05%)
	
	Epoch 2:
	Loss=1.1906986236572266 Batch_id=97 LR=0.00990 Accuracy=53.12: 100%|██████████| 98/98 [00:40<00:00,  2.45it/s]
	
	Test set: Average loss: 0.0037, Accuracy: 4405/10000 (44.05%)
	
	Epoch 3:
	Loss=0.9887471199035645 Batch_id=97 LR=0.01780 Accuracy=61.33: 100%|██████████| 98/98 [00:40<00:00,  2.41it/s]
	
	Test set: Average loss: 0.0031, Accuracy: 5421/10000 (54.21%)
	
	Epoch 4:
	Loss=0.8137422204017639 Batch_id=97 LR=0.02416 Accuracy=68.62: 100%|██████████| 98/98 [00:41<00:00,  2.39it/s]
	
	Test set: Average loss: 0.0015, Accuracy: 7332/10000 (73.32%)
	
	Epoch 5:
	Loss=0.7336296439170837 Batch_id=97 LR=0.02656 Accuracy=73.65: 100%|██████████| 98/98 [00:41<00:00,  2.37it/s]
	
	Test set: Average loss: 0.0016, Accuracy: 7394/10000 (73.94%)
	
	Epoch 6:
	Loss=0.6894105076789856 Batch_id=97 LR=0.02626 Accuracy=77.93: 100%|██████████| 98/98 [00:41<00:00,  2.38it/s]
	
	Test set: Average loss: 0.0016, Accuracy: 7388/10000 (73.88%)
	
	Epoch 7:
	Loss=0.6051175594329834 Batch_id=97 LR=0.02540 Accuracy=80.25: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0014, Accuracy: 7713/10000 (77.13%)
	
	Epoch 8:
	Loss=0.4582046866416931 Batch_id=97 LR=0.02401 Accuracy=82.53: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0010, Accuracy: 8267/10000 (82.67%)
	
	Epoch 9:
	Loss=0.5204144716262817 Batch_id=97 LR=0.02215 Accuracy=84.24: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0009, Accuracy: 8542/10000 (85.42%)
	
	Epoch 10:
	Loss=0.4067685604095459 Batch_id=97 LR=0.01990 Accuracy=86.08: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0009, Accuracy: 8467/10000 (84.67%)
	
	Epoch 11:
	Loss=0.3895864486694336 Batch_id=97 LR=0.01736 Accuracy=87.25: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0009, Accuracy: 8631/10000 (86.31%)
	
	Epoch 12:
	Loss=0.30533018708229065 Batch_id=97 LR=0.01464 Accuracy=88.89: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0009, Accuracy: 8595/10000 (85.95%)
	
	Epoch 13:
	Loss=0.3287360370159149 Batch_id=97 LR=0.01186 Accuracy=90.41: 100%|██████████| 98/98 [00:41<00:00,  2.35it/s]
	
	Test set: Average loss: 0.0007, Accuracy: 8873/10000 (88.73%)
	
	Epoch 14:
	Loss=0.27418187260627747 Batch_id=97 LR=0.00915 Accuracy=91.55: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0007, Accuracy: 8880/10000 (88.80%)
	
	Epoch 15:
	Loss=0.24516217410564423 Batch_id=97 LR=0.00662 Accuracy=92.69: 100%|██████████| 98/98 [00:41<00:00,  2.35it/s]
	
	Test set: Average loss: 0.0006, Accuracy: 8991/10000 (89.91%)
	
	Epoch 16:
	Loss=0.2548474073410034 Batch_id=97 LR=0.00437 Accuracy=93.93: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0006, Accuracy: 9122/10000 (91.22%)
	
	Epoch 17:
	Loss=0.13494442403316498 Batch_id=97 LR=0.00252 Accuracy=94.97: 100%|██████████| 98/98 [00:41<00:00,  2.35it/s]
	
	Test set: Average loss: 0.0006, Accuracy: 9149/10000 (91.49%)
	
	Epoch 18:
	Loss=0.09630003571510315 Batch_id=97 LR=0.00114 Accuracy=95.65: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0006, Accuracy: 9151/10000 (91.51%)
	
	Epoch 19:
	Loss=0.10515687614679337 Batch_id=97 LR=0.00028 Accuracy=96.02: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0006, Accuracy: 9186/10000 (91.86%)
	
	Epoch 20:
	Loss=0.09443861991167068 Batch_id=97 LR=0.00000 Accuracy=96.28: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s]
	
	Test set: Average loss: 0.0006, Accuracy: 9197/10000 (91.97%)
~~~

