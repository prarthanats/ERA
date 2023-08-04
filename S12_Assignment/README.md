# Custom Resnet on CIFAR10 using pytorch Lightening and GradCAM

This repository contains an application for CIFAR-10 classification using PyTorch Lightning. Image Classification is implemented using custom Resnet. The Application includes functionalities for missclassification and GradCam

##Requirements

1. Use the Custom ResNet architecture for CIFAR10 from Assignment 10:
	''' 
	
		PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
		Layer1 -
		X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
		R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
		Add(X, R1)
		Layer 2 -
		Conv 3x3 [256k]
		MaxPooling2D
		BN
		ReLU
		Layer 3 -
		X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
		R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
		Add(X, R2)
		MaxPooling with Kernel Size 4
		FC Layer 
		SoftMax
	
	'''
2. Uses One Cycle Policy such that:
	'''
	
		Total Epochs = 24
		Max at Epoch = 5
		LRMIN = FIND
		LRMAX = FIND
		NO Annihilation
		
	'''

3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
4. Batch size = 512
5. Use ADAM, and CrossEntropyLoss
6. Target Accuracy: 90%

## General Requirements

Spaces app has these features:
1. ask the user whether he/she wants to see GradCAM images and how many, and from which layer, allow opacity change as well
2. ask whether he/she wants to view misclassified images, and how many
3. allow users to upload new images, as well as provide 10 example images
4. ask how many top classes are to be shown (make sure the user cannot enter more than 10)

## Introduction 
### CIFAR Data
The CIFAR-10 dataset consists of 60000 32x32 RGB colour images  each of size 32x32 pixels, in 10 classes. There are 50000 training images and 10000 test images. Analysis on the dataset can be found here. 

1. Images are equally distributed across classes, no class imbalance
2. The 10 classes in CIFAR-10 are:


3. Mean and Standard Deviation for the CIFAR Data is 

'mean [0.49139968 0.48215841 0.44653091]'
'standard deviation [0.24703223 0.24348513 0.26158784]'

4. The dataset contains 10 classes, below are 10 sample images from each class, 

we can see that some of the classes in automobile have gray scale
Also the last image of aeroplane and bird look similar

###PyTorch Lightning

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

## Notebook
The notebook for this assignment can be accessed here:  

### Model Architecture


### Model Summary


## Implementation and Inference Details

~~~
	Epochs - 24
	Batch Size - 512
	Number of parameters: 6,573,130 parameters
	Best Training Accuracy - 98.84% (24th Epoch)
	Best Testing Accuracy - 92.33% (24th Epoch)
	LR Scheduler: OneCycleLR with pct_start = 0.2 (~5/24) since max_lr is required at Epoch 5, out of 24 total epochs
	Optimizer - Adam Scheduler 
	
~~~

Accuracy Metric
The model uses the Accuracy metric from the torchmetrics library to track accuracy during training and validation. It is a multiclass accuracy metric with 10 classes corresponding to the CIFAR-10 categories.

Training, Validation, and Testing Steps
The class defines the training_step, validation_step, and test_step methods to perform the forward pass, calculate the loss, and update the accuracy metric during training, validation, and testing, respectively.

Optimizer and Learning Rate Scheduler
The model uses the Adam optimizer with weight decay for parameter optimization. It also utilizes the OneCycleLR learning rate scheduler to automatically adjust the learning rate during training.

Data Loading
The class provides methods for preparing and setting up the CIFAR-10 dataset for training, validation, and testing. It also includes a method, collect_misclassified_images, for collecting misclassified images during testing for further analysis.

Visualization
The class includes methods, show_misclassified_images and get_gradcam_images, for visualizing misclassified images and GradCAM visualizations of misclassified images, respectively.

### Accuracy and Loss Plots


### Misclassified Images and GradCAM Images


### Class Level Accuracy


### Training Log
