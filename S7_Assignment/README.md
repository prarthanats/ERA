# MNIST Digit Classification 

In this assignment, we will be using the MNIST data for classifying handwritten digits using fully convolutional model. 

## MNIST Data
The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems.The database is also widely used for training and testing in the field of machine learning. It is made up of a number of grayscale pictures that represent the digits 0 through 9. The collection contains square images that are each 28x28 pixels in size, for a total of 784 pixels per image.

![download](https://github.com/prarthanats/ERA/assets/32382676/6fb2614b-569e-4a7b-aaff-9877fdb530b3)

## Requirements

1. 99.4%(this must be consistently shown in your last few epochs, and not a one-time achievement)
2. Less than or equal to 15 epochs
3. Less than 8000 Parameters

## General Requirements

1. Using modular code
2. Exactly 3 steps
3. Each File must have a "target, result, analysis" 

## Block 1 - Get the structure right

Tried to implement a Squeeze-and-Excitation network to classify the numbers in MINIST dataset. This architecute consist of convolution blocks followed by transition blocks. In the first block the main idea is to get the skeleton correct and try and reduce the parameters to around 8000. Dataloader are defined, train and test transform are set. 

The MNIST Data is best represented at edges. the implemented structure is designed to extract edges at the Receptive Fields of 5. Padding of 1 is added to the first convolution block. Even though it is not very helpfull during the initial models. It is added beacuse we are do image augmentations at the 3rd block and padding can help ensure that the rotated image fits within the desired dimensions.

### Model 1 - Skeleton 
#### Target:
1. To get the basic skeleton correct
2. A modular code

#### Results:
1. Total parameters: 75,408
2. Best Training Accuracy - 99.33 at the 15th Epoch
3. Best Testing Accuracy - 99.01 at the 14th epoch

![Model1](https://github.com/prarthanats/ERA/assets/32382676/e54bc625-4f07-4fac-bf00-1cb6e9c7c5df)

#### Analysis:
1. Extremely Heavy Model for such a problem
2. Train and test discrepancy is around 0.30 at the last 6 epochs, and it seems to be consistent, model is not generalizing well on the test data
3. Does indicate overfitting

<img width="580" alt="model1" src="https://github.com/prarthanats/ERA/assets/32382676/25bbb8fa-7df6-4aff-be32-b44be4f72b54">

### model 2 - Less Parameters
#### Target:
1. Same Skeleton as model 1
2. Make the model lighter by reducing the number of channels across all layers

#### Results:
1. Total parameters: 14,760
2. Best Training Accuracy - 98.53 at the 15th Epoch
3. Best Testing Accuracy - 98.55 at the 10th epoch

![model2](https://github.com/prarthanats/ERA/assets/32382676/b79e0920-f8fc-4cb5-8a60-b8e70d03066a)

#### Analysis:
1. Good model, the train and test accuracys are overlapping at a few places. 
2. Train and test discrepancy is less, average is around 0.2 but not consistent and is around 0.3 at the last epoch, so its still overfitting but not as model1
3. The model is capable and can still be pushed to work with lesser parameters

<img width="571" alt="model2" src="https://github.com/prarthanats/ERA/assets/32382676/805cde14-1ead-4496-866d-7fbfd41a8ced">

### model 3 - Lesser Parameters

#### Target:
1. Same Skeleton as model 1
2. Make the model lighter by reducing the number of channels across the model for the assignment requirement

#### Results:
1. Total parameters: 9,446
2. Best Training Accuracy - 98.82 at the 15th Epoch
3. Best Testing Accuracy - 98.82 at the 14th epoch

![model2](https://github.com/prarthanats/ERA/assets/32382676/e8ad01e2-a7ae-4c71-85bc-504107aba8e9)


#### Analysis:
1. Good model
2. Train and test discrepancy is less but is not consistent and around the last epoch the model looks to be little overfitting
3. The model is better,and much lighter than the earlier models, very less overfitting. 
4. we can proceed with this particular skeleton.

![image](https://github.com/prarthanats/ERA/assets/32382676/53bc1d5f-16e4-4eef-b34a-62fe9ddbdb64)

At the end of Block one we have a 9k Parameter model, which is good and not overfitting a lot. The next step is to reduce the overfitting, improve efficency

 
## Block 2 - Improve efficiency of Model

Techniques like dropout,batch normalization are used to reduce the overfitting by adding constraints or introducing randomness during training also adjusting the model architecture to prevent it from overfitting.

### model 4 - Batch Normalization added to every convolution layer except last one 

#### Target:
1. Make the model lighter 
2. Add Batch normalization to increase model efficiency and improve accuracy
#### Results:
1. Total parameters: 8,714
2. Best Training Accuracy - 99.41 at the 15th Epoch
3. Best Testing Accuracy - 99.29 at the 14th epoch

![model4](https://github.com/prarthanats/ERA/assets/32382676/099d573a-3309-46f8-b399-b797edd235a3)

#### Analysis:
1. Train and test discrepancy is less and is around 0.12 and model seems to be good. 
2. Accuracy of train and test has increased after applying batch normalization. the testing accuracy has increased from 98.82 to 99.29 and train accuracy from 98.82 to 99.41 
3. Less overfitting and hence can be altered for better accuracy
4. Parameters are also reduced

![model4](https://github.com/prarthanats/ERA/assets/32382676/2de09512-d50d-40f7-89fb-1f54c5e34b2a)

### model 5 - DropOut of 0.01 is added to every convolution layer except last one 

#### Target:

1. Add Regularization, Dropout

#### Results:
1. Total parameters: 8,694
2. Best Training Accuracy - 99.24 at the 15th Epoch
3. Best Testing Accuracy - 99.22 at the 15th epoch

![model5](https://github.com/prarthanats/ERA/assets/32382676/1192a3d1-9794-4b7a-a3e3-3a6c60e6a25b)

#### Analysis:
1. Good model, Train and test discrepancy is less and seems to be close to zero 
2. The model gap between train and test accuracy has decreased from  0.12 in the previous model to around 0.02
3. Adding dropout has got the test accuracy close to train accuracy, reducing the over fit. hence the accuracy of test has decreased 99.29 in the previous modek to 99.22
4. The train accuracy has also decreased from 99.41 to 99.24. Dropout might be causing underfitting, which might be because of the introduction of noise. 

![image](https://github.com/prarthanats/ERA/assets/32382676/4bdfd45a-a15c-4279-89a6-c8da65863832)


### model 6 - GAP replaces 6*6 Kernel and add a 1*1 in last layer

#### Target:
1. Add GAP and remove the last BIG kernel (6*6 Kernel) 
2. Add a 1*1 in the last layer

#### Results:
1. Total parameters: 5,194
2. Best Training Accuracy - 98.97 at the 15th Epoch
3. Best Testing Accuracy - 99.35 at the 12th epoch

![model6](https://github.com/prarthanats/ERA/assets/32382676/c97097d6-c12d-4b51-a63d-6bbc920c3e2f)

#### Analysis:
1. Total paramters decreased is around 3500 parameters and thats around half the total paramters in the previous model.GAP has reduced the dimensionality of the feature maps resulting in a better representation. 
2. Introducing GAP has included some discrepancy, the model cannot be compared but does seem to be underfitting. 

![image](https://github.com/prarthanats/ERA/assets/32382676/5fa78f3a-caa5-4968-8bc4-c8d7686bff55)

### model 7 - Adding Capacity to the Model 

#### Target:
1. Increasing the number of parameters to the GAP Model
2. Matching the requirements with respect to parameters

#### Results:
1. Total parameters: 7,598
2. Best Training Accuracy - 98.92 at the 15th Epoch
3. Best Testing Accuracy - 99.20 at the 15th epoch

![model7](https://github.com/prarthanats/ERA/assets/32382676/74b057ec-e7fc-4a46-b865-c2f7b42b580c)


#### Analysis:
1. The total parameters is increased by adding more capacity to the model. 
2. The model seems to be efficient. 
3. The model has lesser gap between the train and test accuracy compared to the previous model 
4. We have met some of our objectives in terms of less than 8k parameters, 15 epochs

![image](https://github.com/prarthanats/ERA/assets/32382676/7bf9f9c9-8d06-4207-979d-d101a18e9c95)


At the end of Block two we have a 7k Parameter model, which is good and not overfitting a lot. The next step is to improve accuracy

## Block 3 - Image Augmentation, Learning Rate


### model 8 - Adding Image Augmentation to improve accuracy

#### Target:
1. Inorder to increase the accuracy, Image augmentation can be added

#### Results:
1. Total parameters: 7,598
2. Best Training Accuracy - 98.63 at the 15th Epoch
3. Best Testing Accuracy - 99.26 at the 15th epoch

#### Analysis:
1. Random Rotation between -6.9 and 6.9 degrees. Introducing this variation in the dataset has improved the test accuracy from 99.20 in previous model to 99.26 in this model. 
2. This model seems to be performing consistent for the test data

![model8](https://github.com/prarthanats/ERA/assets/32382676/1201abca-b4f2-44f3-9dab-1ef28039df81)


### model 9 - Adding Image Augmentation with Learning Rate

#### Target:
1. Inorder to increase the accuracy, Learning rate and Image augmentation can be added

#### Results:
1. Total parameters: 7,598
2. Best Training Accuracy - 99.08 at the 15th Epoch
3. Best Testing Accuracy - 99.42 from the 12th epoch

#### Analysis:
1. transforms.RandomAffine(degrees=15) is added as additional augmentation. Introducing this variation in the dataset has improved the test accuracy
2. By adding ReduceLRonPlateau helps the model converge more effieciently to reach a better optima
3. Adding these 2 have increased the model accuracy and more consistent results

![model9](https://github.com/prarthanats/ERA/assets/32382676/13e7a1c3-6d2d-40a2-89e1-c6a4d6c20c56)


### model 10 - Reduced parameters on Image Augmentation with Learning Rate

#### Target:
1. Reduced parameters to work with lesser parameters

#### Results:
1. Total parameters: 6,022
2. Best Training Accuracy - 98.86 at the 15th Epoch
3. Best Testing Accuracy - 99.41 from the 14th epoch

![model10](https://github.com/prarthanats/ERA/assets/32382676/187a91ff-885f-4672-b2aa-31d211866060)


#### Analysis:
1. This model has been more of an experiment to see what is the right parameter size with which this model to could give atleast 2 test accuracy more that 99.4.
2. With around 6k parameters, the model seems to be lineraly increasing the test and train accuracy. 

![model10](https://github.com/prarthanats/ERA/assets/32382676/a1658ff2-19ab-4eb1-9584-7a07bcaed54a)

