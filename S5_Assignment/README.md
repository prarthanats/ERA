<h1> Implementation of Digit Classification on MNIST Data using PyTorch </h1>

This repository is an implementation of training and testing code for digit classification on MNIST data using PyTorch.

<h2> MNIST Data </h2>
The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems.The database is also widely used for training and testing in the field of machine learning. It is made up of a number of grayscale pictures that represent the digits 0 through 9. The collection contains square images that are each 28x28 pixels in size, for a total of 784 pixels per image.

<h2> Requirements </h2>
Python>=3.5
<br>
torch=2.0.1
</br>
torchvision=0.15.2

<h2> How to use it? </h2> 

<h2> Digit Classification on MNIST Data </h2>

<h3> model.py </h3>
- This file contains the model implemented using Convolution layers and Fully connected layers. 
- 
<h3> utils.py </h3>
- get_trainloader() - to get the training data, transform it and load it iteratively
- get_testloader() - to get the test data, transform it and load it iteratively
- GetCorrectPredCount() - Calculates the count of correct predictions given predicted values and corresponding labels.
- train() - model training, performs a forward pass to get the prediction, calculates loss, backpropogates, updates and tracks accuracy and loss
- test()- Evaluates model on the test data, calculates the test loss between prediction and actual labels and tracks accuracy and loss

<h3> main_mnsit.py </h3>
- main function for each epoch runs the train and test data and calculates the accuracy, and provides the model summary

Resuts on MNIST dataset

