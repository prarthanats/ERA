# Custom Resnet on CIFAR10 using pytorch Lightening and GradCAM

This repository contains an application for CIFAR-10 classification using PyTorch Lightning. Image Classification is implemented using custom Resnet. The Application includes functionalities for missclassification and GradCam

## Requirements

1. Use the Custom ResNet architecture for CIFAR10 from Assignment 10:
2. Uses One Cycle Policy 
3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
4. Batch size = 512 and Use ADAM, and CrossEntropyLoss
5. Target Accuracy: 90%

## General Requirements

Spaces app needs to include these features:
1. ask the user whether he/she wants to see GradCAM images and how many, and from which layer, allow opacity change as well
2. ask whether he/she wants to view misclassified images, and how many
3. allow users to upload new images, as well as provide 10 example images
4. ask how many top classes are to be shown (make sure the user cannot enter more than 10)

## Introduction 

### CIFAR Data
The CIFAR-10 dataset consists of 60000 32x32 RGB colour images  each of size 32x32 pixels, in 10 classes. There are 50000 training images and 10000 test images. Analysis on the dataset can be found here. 
1. Images are equally distributed across classes, no class imbalance
2. The 10 classes in CIFAR-10 are:

![image](https://github.com/prarthanats/ERA/assets/32382676/30df5d06-4055-4f37-88be-bf45816c6f25)

The dataset contains 10 classes, below are 10 sample images from each class, we can see that some of the classes in automobile have gray scale and also the last image of aeroplane and bird look simila
3. Mean and Standard Deviation for the CIFAR Data is 'mean [0.49139968 0.48215841 0.44653091]' and 'standard deviation [0.24703223 0.24348513 0.26158784]'

### PyTorch Lightning

PyTorch Lightning is a lightweight PyTorch wrapper that simplifies the training and organizing of deep learning models. It provides a high-level interface for PyTorch that abstracts away the boilerplate code typically required for training, validation, and testing loops. With PyTorch Lightning, you can focus more on designing your models and less on the repetitive tasks surrounding the training process.

#### Key Features
1. Easy-to-use: PyTorch Lightning simplifies the training process by abstracting away the complexities of the training loop. This allows you to define your model as a PyTorch Lightning module, and the training loop, validation loop, and testing loop are automatically handled for you.
2. Modular Design: With PyTorch Lightning, you can easily organize your code into separate modules, such as data loaders, models, and optimizers, making it more maintainable and scalable.
3. Standardized Interfaces: PyTorch Lightning enforces a standardized interface for training, validation, and testing, making it easier to collaborate with others and integrate with different research projects.
4. Reproducibility: By using PyTorch Lightning, you can achieve better experiment reproducibility with the help of built-in seed handling and deterministic execution.
5. Integration with TensorBoard: PyTorch Lightning provides seamless integration with TensorBoard for easy visualization and monitoring of training metrics.
6. Support for Multiple Hardware Configurations: PyTorch Lightning enables training on multiple GPUs and distributed systems out of the box.
7. Advanced Features: PyTorch Lightning comes with advanced features, such as automatic precision training (16-bit mixed precision), gradient accumulation, and early stopping.

#### Getting Started

To start using PyTorch Lightning, follow these simple steps:

1. Install PyTorch Lightning and its dependencies:

~~~
	pip install pytorch-lightning
~~~
2. Define your model as a PyTorch Lightning module, inheriting from pl.LightningModule.
3. Set up your data loaders and Lightning DataModules, inheriting from pl.LightningDataModule.
4. Initialize a pl.Trainer object to configure your training settings.
5. Train your model using the Trainer object's fit method.
6. Monitor and visualize training progress with TensorBoard.

### Gradio for PyTorch Lightning App

Gradio is a user interface (UI) library that makes it easy to create web-based interfaces for machine learning models. When combined with PyTorch Lightning, Gradio allows you to deploy and share your PyTorch Lightning-powered models with a user-friendly web application.

#### Key Features
1. Interactive User Interface: Gradio enables you to build interactive web interfaces for your PyTorch Lightning models, making it simple for users to interact with your models and get real-time predictions.
2. Support for Multiple Input Types: Gradio supports a wide range of input types, including text, images, audio, video, and more, making it versatile for various machine learning tasks.
3. Automatic Data Conversion: Gradio automatically converts the input data from the user interface to the format expected by your PyTorch Lightning model, simplifying the integration process.
4. Easy Deployment: Deploying your PyTorch Lightning model with Gradio requires minimal effort, allowing you to share your models with others quickly.
5. Visualization Tools: Gradio provides visualization tools to display the model's predictions and intermediate outputs, enhancing model interpretability.

#### Getting Started

1. Create a PyTorch Lightning model, following the standard PyTorch Lightning guidelines.
2. Define a function that takes the input from Gradio's interface and returns the model's predictions.
3. Use the gr.Interface class to create the web interface, passing in the function defined 

~~~
	import gradio as gr
	gr_interface = gr.Interface(fn=predict, inputs="text", outputs="text") #Assuming you have a PyTorch Lightning model 'model' and a prediction function 'predict'
	gr_interface.launch()
~~~

The Gradio app will be accessible at the provided URL, and users can now interact with your PyTorch Lightning model via the web interface.

## Model Architecture

The Custom Resnet Model Consists of 3 three classes
~~~
	PrepBlock:
	
	The PrepBlock is the first block in the convolutional neural network. It is responsible for the initial processing of the input images.
	1. A 2D convolutional layer (nn.Conv2d) with 3 input channels (assuming RGB images) and 64 output channels. The kernel size is set to (3, 3), and padding is added to maintain the spatial dimensions of the input.
	2. A ReLU activation function (nn.ReLU) to introduce non-linearity in the model.
	3. A batch normalization layer (nn.BatchNorm2d) to stabilize training and accelerate convergence.
	4. A dropout layer (nn.Dropout) with a specified dropout probability, which helps prevent overfitting during training.

	ConvolutionBlock:
	
	The ConvolutionBlock is a building block for the main part of the convolutional neural network. It performs a series of convolutional operations and pooling.
	The conv module consists of four layers:
	1. A 2D convolutional layer (nn.Conv2d) with a specified number of input channels (in_channels) and output channels (out_channels). The kernel size is set to (3, 3), and padding is added for the same spatial dimensions.
	2. A 2D max-pooling layer (nn.MaxPool2d) with a kernel size of (2, 2) to downsample the spatial dimensions by half.
	3. A batch normalization layer (nn.BatchNorm2d) to normalize the output of the convolutional layer.
	4. A ReLU activation function (nn.ReLU) to introduce non-linearity in the model.

	ResidualBlock:
	
	The ResidualBlock is a variation of the residual block used in ResNet architectures. It introduces shortcut connections to improve gradient flow and mitigate the vanishing gradient problem.
	The residual module consists of six layers:
	1. A 2D convolutional layer (nn.Conv2d) with a specified number of input channels (channels) and output channels (channels). The kernel size is set to (3, 3), and padding is added for the same spatial dimensions.
	2. A batch normalization layer (nn.BatchNorm2d) to normalize the output of the convolutional layer.
	3. A ReLU activation function (nn.ReLU) to introduce non-linearity in the model.
	4. Another 2D convolutional layer (nn.Conv2d) with the same number of input and output channels and the same kernel size and padding.
	5. Another batch normalization layer (nn.BatchNorm2d).
	6. Another ReLU activation function (nn.ReLU).
	
These classes provide fundamental building blocks for creating a convolutional neural network for image classification. You can use them to design more complex network architectures by stacking these blocks together in various configurations.
~~~

### Model Summary

<img width="259" alt="Model_Summary_Lightening" src="https://github.com/prarthanats/ERA/assets/32382676/271deb0f-edba-4187-8330-1c516f473757">

## Implementation and Inference Details
~~~
	Epochs - 24
	Batch Size - 512
	Number of parameters: 6,573,130 parameters
	Best Training Accuracy - 98.84% (24th Epoch)
	Best Testing Accuracy - 92.43% (24th Epoch)
	LR Scheduler: OneCycleLR with pct_start = 0.2 (~5/24) since max_lr is required at Epoch 5, out of 24 total epochs
	Optimizer - Adam Scheduler 
~~~

## Notebook
1. The notebook for this assignment can be accessed here:  [Assignment 12](https://github.com/prarthanats/ERA/blob/main/S12_Assignment/Assignment_12_CustomResnet.ipynb)
2. The CustomResNet Lightening Module includes all the classes including:
~~~
	1. forward - The forward function defines the forward pass of the model. It takes an input tensor x and passes it through the layers of the model to produce the output predictions.
	2. training_step - The training_step function defines the operations performed during a single training step. It takes a batch of training data and computes the loss and any other relevant metrics for that batch.
	3. validation_step - The validation_step function defines the operations performed during a single validation step. It takes a batch of validation data and computes the loss and metrics for that batch.
	4. test_step - The test_step function defines the operations performed during a single test step. It takes a batch of test data and computes the loss and metrics for that batch.
	5. configure_optimizers - The configure_optimizers function defines the optimizer(s) used during training. It allows you to specify the optimization algorithm and its hyperparameters.
	6. prepare_data - The prepare_data function is used for data preparation tasks that need to be performed only once, such as downloading and preprocessing the dataset.
	7. setup - The setup function is used for any setup or initialization tasks that need to be performed on each GPU or distributed process before training begins
	8. train_dataloader - The train_dataloader function defines the data loader for the training dataset. It specifies how training batches are sampled and prepared for training. The same is done for Validation and Test data
	9. show_misclassified_images - The show_misclassified_images function is a custom function that can be used to visualize misclassified images during evaluation.
~~~

3. The Configurations for the notebook is provided, it provides classes, learning rate, batch size, epochs
~~~
	config = {
    'batch_size': 512,
    'data_dir': './data',
    'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'num_classes': 10,
    'lr': 0.01,
    'max_lr': 0.1,
    'max_lr_epoch': 5,
    'dropout' : 0.01,
}
~~~
4. Tensor Board is used with Lightening for visualization and Montioring the training

   ![image](https://github.com/prarthanats/ERA/assets/32382676/6106d433-9f67-4785-a98a-ed57a067e20c)

5. The Model is saved using
~~~
	torch.save(model.state_dict(), "resnet_model_v2.pth")
~~~
6. Gradio is then used for visulaization of the app. The App related information can be found [Custom Resner App](https://huggingface.co/spaces/PrarthanaTS/Cifar10)


### Accuracy and Loss Plots


### Misclassified Images and GradCAM Images


### Class Level Accuracy


### Training Log
