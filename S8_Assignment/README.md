# Application of Normalization on CIFAR 10 Dataset 

In this assignment we will be implementing the different normalization techniques such as Batch, Layer and Group on the CIFAR 10 dataset using PyTorch.

## Requirements

1. Make a model using CIFAR dataset

	Network with Group Normalization
	Network with Layer Normalization
	Network with Batch Normalization
	
2. Less than 50000 parameters
3. Find 10 misclassified images for each of the 3 models, and show them as a 5x2 image matrix in 3 separately annotated images
4. More than 70% accuracy
5. 20 epochs

## General Requirements

1. Using modular code

## Introduction to CIFAR Data

The CIFAR-10 dataset consists of 60000 32x32 RGB colour images  each of size 32x32 pixels, in 10 classes. There are 50000 training images and 10000 test images. Analysis on the dataset can be found here. 

1. Images are equally distributed across classes, no class imbalance
2. The 10 classes in CIFAR-10 are:

    It can be seen that some of the classes in automobile have gray scale. Also the last image of aeroplane and bird look similar 	
   
![Data_Analysis](https://github.com/prarthanats/ERA/assets/32382676/f24c0379-4f06-4a31-8a91-184499e677f4)

3. Mean and Standard Deviation for the CIFAR Data is $0.49139968 0.48215841 0.44653091$ and $0.24703223 0.24348513 0.26158784$

## Normalization
Normalization are techniques used in deep learning to normalize the activations of neurons in a neural network. They help address the issue of internal covariate shift, which refers to the change in the distribution of input values to a layer during training.Batch Normalization, Layer Normalization, and Group Normalization are the techniques used to address the internal covariate shift but they differ in how they normalize the activations and the level at which normalization is applied.

1. Batch Normalization (BN)

The statistics (mean and variance) are computed across the batch and the spatial dimensions
Rescaling the data points w.r.t each channel
It computes the mean and variance across the mini-batch for each channel and normalizes the activations based on these statistics
It introduces learnable scale and shift parameters to allow the network to adapt the normalized activations
It reduces the dependence of gradients on the scale of the parameters or of their initial values. This allows us to use much higher learning rates
It is effective for training deep networks, especially in tasks such as image classification

![batch-norm](https://github.com/prarthanats/ERA/assets/32382676/15a7ea5e-085f-46f5-ae45-8a6c8c311320)

2. Layer Normalization (LN)

The statistics (mean and variance) are computed across all channels and spatial dims
Rescaling the data points w.r.t each image across all channels
It computes the mean and variance across the spatial dimensions for each channel and normalizes the activations based on these statistics
It operates on the spatial dimensions (e.g., height and width) of the activations within a layer
It is commonly used in recurrent neural networks (RNNs) and natural language processing tasks

![layer-norm](https://github.com/prarthanats/ERA/assets/32382676/100869f5-a0de-4b9d-b6fe-adc73de47f24)

3. Group Normalization (GN)

It divides the channels into groups and computes the first-order statistics within each group
Rescaling the data points w.r.t specific group of layer in an image
It operates on both the channel and spatial dimensions of the activations. 
It divides the channels into groups and computes the mean and variance separately for each group.
It normalizes the activations within each group based on their group-specific statistics.
It can be effective when batch size is small such as when processing multiple images or samples with different characteristics. 
GN introduces additional parameters for scale and shift, similar to BN.

![group-normalization](https://github.com/prarthanats/ERA/assets/32382676/24dd445f-c2d3-4a0d-a9d0-53c9cae1b687)

The choice of normalization method depends on the specific task, network architecture, and the available data.

## Experiments and Model performance wrt Normalization

|Normalization |Batch size |Dropout |Parameters |Best Train Accuracy |Best Test Accuracy | Link |
|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| Batch Normalization | 128 | 0.01 | 41,992 | 82.31 | 76.47 | |
| Layer Normalization | 128 | 0.01 | 41,992 | 75.08 | 71.67 | |
| Group Normalization | 128 | 0.01 | 41,992 | 76.00 | 72.14 | |
| Batch Normalization with skip connection | 128 | 0.01 | 32,584 | 80.13 | 74.97 | |
| Group Normalization with lesser batch | 64 | 0.01 | 41,992 | 75.87 | 72.25 | |


Observations

1. Batch normalization performs better than other two techniques with a highest accuracy of 76.47
2. Batch normalization performs better with higher batch sizes as the statistics are computed over a larger number of samples
3. Layer normalization performance was lower compared to other two techniques. Can handle samples independently, making it more suitable for tasks involving sequential data
4. Group Normalization performs better with lower batchsize, with a batch size of 128, earier epoch performed better but with a batch size of 64 it performs better at higher epochs
5. Batch normalization with skip added at convolution layer 6, gave a similar accuracy with 9k less parameters


Receptive Field Calculation


About Code

The final code can be found here 

Code for individual experiments for each normalization technique can be found here
We have used modularized structure for this assignment by creating few utility funtions, the details for this can be found here

1. The code is modularized and has seperate functions
2. Net function takes GN/LN/BN to decide which normalization to include and creates the network architecture accordingly
3. We have a single model.py file that includes norm_layer function that takes either GN/LN/BN to pass the parameters. 
4. model.py file also includes a training function and testing function
5. Visualize.py have a function to plot the metrics, print missclassified images
6. Dataloader function downloads, transform the data

Validation Accuracy and Loss




Misclassified Images

Batch Normalization (GN)



Layer Normalization (LN)



Group Normalization (GN)
