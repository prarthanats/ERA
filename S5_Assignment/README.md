# Implementation of Digit Classification on MNIST Data using PyTorch

This repository is an implementation of training and testing code for digit classification on MNIST data using PyTorch.

## MNIST Data
The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems.The database is also widely used for training and testing in the field of machine learning. It is made up of a number of grayscale pictures that represent the digits 0 through 9. The collection contains square images that are each 28x28 pixels in size, for a total of 784 pixels per image.

## Requirements
Python>=3.5
<br>
torch=2.0.1
</br>
torchvision=0.15.2

## Digit Classification on MNIST Data
### Model ([model.py](https://github.com/prarthanats/ERA/blob/main/S5_Assignment/model.py))
- This file contains the model implemented using Convolution layers and Fully connected layers. The model is defined in the `Net` class, which inherits from the `nn.Module` base class
- We are using 4 convolution layers with 2 max pooling between them, Relu is used as activation function to keep the values positive, except in the last layer
  - Convolution filters an image for a particular feature
  - ReLU detects that feature within the filtered image
  - Maximum pooling condenses the image to enhance the features
  - Two fully connected Layers are used once the data is flattened

![TorchView](https://github.com/prarthanats/ERA/assets/32382676/d17be825-583c-433c-a8b9-64e282b4a432)

### Utils ([utils.py](https://github.com/prarthanats/ERA/blob/main/S5_Assignment/utils.py))
- For the training, transformations such as scaling, normalizing, cropping, and flipping. This will help the network generalize the model leading to a better performance. The input data is resized to 28x28 pixels. The testing data are used to check the model's performance on data. For this noscaling or rotation transformations are required.
- get_trainloader() - to get the training data, transform it and load it iteratively
- get_testloader() - to get the test data, transform it and load it iteratively
- GetCorrectPredCount() - Calculates the count of correct predictions given predicted values and corresponding labels.
- train() - model training, performs a forward pass to get the prediction, calculates loss, backpropogates, updates and tracks accuracy and loss
- test()- Evaluates model on the test data, calculates the test loss between prediction and actual labels and tracks accuracy and loss
- Plotting the training and test accuracy/loss

### Main ([MNIST_Handwritten_Digit_Classification_using_Convolution_Neural_Network__S5.py](https://github.com/prarthanats/ERA/blob/main/S5_Assignment/MNIST_Handwritten_Digit_Classification_using_Convolution_Neural_Network__S5.ipynb))
- Checks if CUDA is available and set the device accordingly
- Load the train and test data from the utils
- Load the data visulaization, train and test
- for each epoch runs the train and test functions and calculates the accuracy and loss, and provides the model summary

## Result on MNIST Data
![accuracy_loss](https://github.com/prarthanats/ERA/assets/32382676/c8bb7800-016b-4282-b40a-cd61ba607220)
