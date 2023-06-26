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
5. Less than 20 epochs

## General Requirements

1. Using modular code

## Introduction to CIFAR Data

The CIFAR-10 dataset consists of 60000 32x32 RGB colour images  each of size 32x32 pixels, in 10 classes. There are 50000 training images and 10000 test images. Analysis on the dataset can be found here. [Data Analysis](https://github.com/prarthanats/ERA/blob/main/S8_Assignment/data_analysis/data_analysis.ipynb)

1. Images are equally distributed across classes, no class imbalance
2. The 10 classes in CIFAR-10 are:

    It can be seen that some of the classes in automobile have gray scale. Also the last image of aeroplane and bird look similar 	
   
![Data_Analysis](https://github.com/prarthanats/ERA/assets/32382676/f24c0379-4f06-4a31-8a91-184499e677f4)

3. Mean and Standard Deviation for the CIFAR Data is $0.49139968, 0.48215841, 0.44653091$ and $0.24703223, 0.24348513, 0.26158784$

## Normalization
Normalization are techniques used in deep learning to normalize the activations of neurons in a neural network. They help address the issue of internal covariate shift, which refers to the change in the distribution of input values to a layer during training.Batch Normalization, Layer Normalization, and Group Normalization are the techniques used to address the internal covariate shift but they differ in how they normalize the activations and the level at which normalization is applied.

|Normalization |Computation | Rescaling | Description | Works Best|Representation |
|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
|Batch Normalization (BN) |Computes the mean and variance across the mini-batch for each channel and normalizes the activations based on these statistics|Rescaling the data points w.r.t channel|It introduces learnable scale and shift parameters to allow the network to adapt the normalized activations|It is effective for training deep networks, especially in tasks such as image classification|![batch-norm](https://github.com/prarthanats/ERA/assets/32382676/15a7ea5e-085f-46f5-ae45-8a6c8c311320)|
|Layer Normalization (LN) |Computes the mean and variance across the spatial dimensions for each channel and normalizes the activations based on these statistics| Rescaling the data points w.r.t image across all channels|It operates on the spatial dimensions (e.g., height and width) of the activations within a layer|It is commonly used in recurrent neural networks (RNNs) and natural language processing tasks|![layer-norm](https://github.com/prarthanats/ERA/assets/32382676/100869f5-a0de-4b9d-b6fe-adc73de47f24)|
|Group Normalization (GN)|It divides the channels into groups and computes the first-order statistics within each group|Rescaling the data points w.r.t specific group of layer in an image|It normalizes the activations within each group based on their group-specific statistics.It operates on both the channel and spatial dimensions of the activations|It can be effective when batch size is small such as when processing multiple images or samples with different characteristics. |![group-normalization](https://github.com/prarthanats/ERA/assets/32382676/24dd445f-c2d3-4a0d-a9d0-53c9cae1b687)|


The choice of normalization method depends on the specific task, network architecture, and the available data.

## Experiments and Model performance wrt Normalization

|Normalization |Batch size |Dropout |Parameters |Best Train Accuracy |Best Test Accuracy | Observation  |Link |
|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| Batch Normalization | 128 | 0.01 | 41,992 | 82.31 | 76.47 | Batch normalization performs better than other two techniques with a highest accuracy of 76.47. Batch normalization performs better with higher batch sizes | [BN](https://github.com/prarthanats/ERA/blob/main/S8_Assignment/Normalization_Experiment/Batch_Normalization.ipynb) |
| Layer Normalization | 128 | 0.01 | 41,992 | 75.08 | 71.67 | Layer normalization accuracy was lower compared to other two techniques. In the initial epochs LN had higher accuracy than GN but at later epochs GN had better accuracy. LN has slightly lower loss |[LN](https://github.com/prarthanats/ERA/blob/main/S8_Assignment/Normalization_Experiment/Layer_Normalization.ipynb)|
| Group Normalization | 128 | 0.01 | 41,992 | 76.00 | 72.14 | Group Normalization at initial epochs has a lower accuracy but at later epochs GN works better|[GN](https://github.com/prarthanats/ERA/blob/main/S8_Assignment/Normalization_Experiment/Group_Normalization.ipynb) |
| Batch Normalization with skip connection | 128 | 0.01 | 32,584 | 80.13 | 74.97 | Batch normalization with skip added at convolution layer 6, gave a similar accuracy with 9k less parameters(i had to keep the channel size same for conv5 and conv6)|[BN_skip](https://github.com/prarthanats/ERA/blob/main/S8_Assignment/Normalization_Experiment/Batch_Normalization_xplustry%20(1).ipynb) |
| Group Normalization with lesser batch | 64 | 0.01 | 41,992 | 75.87 | 72.25 |GN performs better with lower batchsize. GN with higher batch size works better initially but with lower batch size works better at final epochs |[GN_64](https://github.com/prarthanats/ERA/blob/main/S8_Assignment/Normalization_Experiment/Group_Normalization_less_batch.ipynb) |

## Receptive Field Calculation

![image](https://github.com/prarthanats/ERA/assets/32382676/bd200d32-d3b9-4e78-82e6-d0b253ed08a2)

## About Code

The final code can be found here [CIFAR with Normalization](https://github.com/prarthanats/ERA/blob/main/S8_Assignment/CIFAR_10_Classification.ipynb)

Code for individual experiments for each normalization technique can be found here [Individual Files](https://github.com/prarthanats/ERA/tree/main/S8_Assignment/Normalization_Experiment)

The details for this can be found here

1. The code is modularized and has seperate functions
2. Net function takes GN/LN/BN to decide which normalization to include and creates the network architecture accordingly [model](https://github.com/prarthanats/ERA/blob/main/S8_Assignment/model.py)
3. We have a single model.py file that includes norm_layer function that takes either GN/LN/BN to pass the parameters. 
4. model.py file also includes a training function and testing function
5. Visualize.py have a function to plot the metrics, print missclassified images [visulaize](https://github.com/prarthanats/ERA/blob/main/S8_Assignment/visualize.py)
6. Dataloader function downloads, transform the data [dataloader](https://github.com/prarthanats/ERA/blob/main/S8_Assignment/dataloader.py)

## Validation Accuracy and Loss

The Accuracy and Loss for BN, LN and GN can be found here,
![Accuracy](https://github.com/prarthanats/ERA/assets/32382676/4fc2809a-9982-412c-9864-fb594ac6521c) ![loss](https://github.com/prarthanats/ERA/assets/32382676/5554fc7b-f7b4-4945-8176-6df42895607a)

## Misclassified Images

### Batch Normalization (BN)

![bn2](https://github.com/prarthanats/ERA/assets/32382676/b9981775-4eba-42aa-ac16-13253db13bac)

### Layer Normalization (LN)

![ln2](https://github.com/prarthanats/ERA/assets/32382676/54e74cbd-bd5c-4de3-bdd1-64e4bd9a6976)

### Group Normalization (GN)

![gn2](https://github.com/prarthanats/ERA/assets/32382676/b5d10fa8-f820-421f-a82a-92097f4bc3bf)

## Model Summary Can be found here

### Batch Normalization (BN)

![bn3](https://github.com/prarthanats/ERA/assets/32382676/86b56488-f23c-4997-af54-f02dd7677591)

### Layer Normalization (LN)

![ln3](https://github.com/prarthanats/ERA/assets/32382676/0cefe6d5-f294-4e1b-9e90-209e86428c7c)

### Group Normalization (GN)

![gn3](https://github.com/prarthanats/ERA/assets/32382676/2112ce03-19e8-436a-9bfb-0f59a931cd4d)

