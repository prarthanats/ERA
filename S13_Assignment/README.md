# Custom YOLOv3 on Pascal VOC using pytorch Lightening

This repository contains an application for Pascal VOC Object detection using PyTorch Lightning. Object detection is implemented using custom YOLOv3. The Application includes functionalities for GradCam

## Requirements

1. Use the Custom Yolov3 architecture for Pascal VOC provided 
2. Move the code to PytorchLightning
3. Train the model to reach such that all of these are true:
~~~
	Class accuracy is more than 75%
	No Obj accuracy of more than 95%
	Object Accuracy of more than 70% (assuming you had to reduce the kernel numbers, else 80/98/78)
	Ideally trailed till 40 epochs
	
~~~
4. Additional Features:
~~~
	Add multi-resolution training - the code shared trains only on one resolution 416
	Add Implement Mosaic Augmentation only 75% of the times
	Train on float16
	GradCam must be implemented
~~~

## Introduction 

### The PASCAL Visual Object Classes Data

The Pascal VOC dataset is a benchmark dataset for object detection and classification tasks. It contains images annotated with bounding boxes and labels for a variety of object classes. The dataset is commonly used to evaluate the performance of object detection models.

#### Dataset Overview
Classes: The dataset includes annotations for 20 object classes, ranging from animals to vehicles and everyday objects.
Annotations: Each image is annotated with bounding boxes around objects of interest and their corresponding class labels.

#### Classes 
~~~
	Person: person
	Animal: bird, cat, cow, dog, horse, sheep
	Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
	Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor
~~~
### YOLO: Real-Time Object Detection

YOLO (You Only Look Once) is an advanced object detection algorithm known for its speed and accuracy. Unlike traditional object detection methods that involve sliding windows or region proposals, YOLO processes the entire image at once to predict object classes and bounding box coordinates directly.

Key Features

1. One Pass Detection: YOLO is designed to detect objects in a single pass through the neural network. It divides the image into a grid and predicts bounding boxes, class probabilities, and objectness scores for each grid cell.
2. Speed: YOLO is exceptionally fast compared to other object detection methods. It can process images in real-time, making it suitable for applications requiring low latency.
3. End-to-End: YOLO performs detection and classification in a single step. It predicts both class probabilities and bounding box coordinates without the need for multiple stages.
4. Multi-Scale Detection: YOLO can detect objects of various sizes and scales within an image. It uses anchor boxes to predict multiple bounding box shapes for different object types.

Usage
To use YOLO for real-time object detection, you can follow these steps:
1. Model Selection: Choose a YOLO variant that suits your needs. YOLO has different versions (e.g., YOLOv3, YOLOv4) with varying trade-offs between speed and accuracy.
2. Image Preprocessing: Preprocess the input images to match the required format and scale for the YOLO model. This typically involves resizing, normalization, and data transformation.
3. Inference: Use the pre-trained YOLO model to perform inference on the preprocessed images. The model outputs bounding box coordinates, class predictions, and confidence scores.
4. Post-processing: Apply non-maximum suppression (NMS) to remove duplicate or overlapping detections. Filter out low-confidence predictions based on a confidence threshold.
5. Visualization: Visualize the detected objects by overlaying bounding boxes and class labels on the original images.

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

## Notebook
The notebook for this assignment can be accessed here:  

### Model Architecture


### Model Summary


## Implementation and Inference Details



## Accuracy Metric




### Object Detection 


### Training Log
