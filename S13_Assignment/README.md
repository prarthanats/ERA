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

Classes: The dataset includes annotations for 20 object classes, ranging from animals to vehicles and everyday objects.
~~~
	Person: person
	Animal: bird, cat, cow, dog, horse, sheep
	Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
	Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor
~~~

### YOLOV3: Real-Time Object Detection

YOLO (You Only Look Once) is an advanced object detection algorithm known for its speed and accuracy. Unlike traditional object detection methods that involve sliding windows or region proposals, YOLO processes the entire image at once to predict object classes and bounding box coordinates directly.

#### Key Features

1. Anchor Boxes: YOLOv3 uses anchor boxes to improve the accuracy of bounding box predictions. These anchor boxes represent different object shapes and sizes, and the model learns to adjust its predictions based on these anchor boxes.
2. Strided Convolutions: YOLOv3 uses strided convolutions to downsample the feature maps, which helps capture context and information at different scales efficiently. This is crucial for detecting objects of different sizes.
3. Multi-Scale Training: YOLOv3 uses a technique called "multi-scale training" where it is trained on images of different sizes. This helps the model generalize better to objects of various scales during inference.
4. Non-Maximum Suppression (NMS): After object detection, YOLOv3 employs non-maximum suppression to filter out redundant bounding box predictions and keep only the most confident ones.
5. Improved Loss Function: YOLOv3 uses a combination of localization loss, confidence loss, and class loss. This loss function helps in better training and more accurate object localization.

### Gradio for PyTorch Lightning App

Gradio is a user interface (UI) library that makes it easy to create web-based interfaces for machine learning models. When combined with PyTorch Lightning, Gradio allows you to deploy and share your PyTorch Lightning-powered models with a user-friendly web application.

The Gradio app will be accessible at the provided URL, and users can now interact with your PyTorch Lightning model via the web interface.

<img width="871" alt="bla" src="https://github.com/prarthanats/ERA/assets/32382676/825577ee-396d-4565-a538-012f962b677b">

## Notebook
The notebook for this assignment can be accessed here:  [Notebook](https://github.com/prarthanats/ERA/blob/main/S13_Assignment/Final_Code_without_mosaic.ipynb)

### Model Architecture
~~~
  CNNBlock: A building block comprising a convolutional layer, batch normalization, and LeakyReLU activation, used to process image features in convolutional neural networks.
  ResidualBlock: A module containing multiple repetitions of two stacked CNNBlocks, capable of performing residual connections to help in feature extraction and information flow.
  ScalePrediction: Generates scale-specific predictions by employing convolutional layers with varying kernel sizes, aiding in object detection tasks, particularly for the YOLO architecture.
  YOLOv3: A YOLO variant for object detection, integrating various CNN layers, ResidualBlocks, and ScalePredictions to provide multi-scale predictions of object classes and bounding boxes in an image.
~~~

### Model Summary

<img width="196" alt="parameters" src="https://github.com/prarthanats/ERA/assets/32382676/30597175-a47d-41e7-b2a0-63d9de062ce5">

## LR Metrics [lr_metrics](https://github.com/prarthanats/ERA/blob/main/S13_Assignment/utils.py)

~~~
The max LR was found using the inbuilt functionality of lightning module.

class LearningRateFinder:
    def __init__(self, trainer, model):
        self.trainer = trainer
        self.model = model

    def find_and_set_learning_rate(self):
        tuner = Tuner(self.trainer)
        lr_finder = tuner.lr_find(self.model)

        # Plot the learning rate curve and get a suggestion
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Get the suggested new learning rate
        new_lr = lr_finder.suggestion()

        # Update the learning rate in the model's hyperparameters
        self.model.hparams.learning_rate = new_lr

~~~

<img width="438" alt="lr" src="https://github.com/prarthanats/ERA/assets/32382676/42519781-2976-46ee-a179-886f67c3c7f3">

## Mosaic Transformation

Mosaic augmentation is a data augmentation technique commonly used in computer vision tasks, especially for object detection. To make the mosaic transformation apply to 75% of the data

~~~
self.counter = (self.counter + 1) % 4 ,

This means that for 75% of the data samples, the mosaic transformation will be applied, as the counter values 1, 2, and 3 correspond to 75% of the total possibilities (since 1/4 is 25%).
~~~

## Accuracy Metric
Training and Testing Accuracy for 39th Epoch

<img width="360" alt="1" src="https://github.com/prarthanats/ERA/assets/32382676/9b1f11db-4006-4194-a7f6-536820aaf732">

Test Losses

<img width="312" alt="Untitled" src="https://github.com/prarthanats/ERA/assets/32382676/66feef4d-4360-4019-b3d2-7aa8d9f8cb45">


### Object Detection Outputs

For Training Data Output

![40th epoch](https://github.com/prarthanats/ERA/assets/32382676/d93a94aa-85a8-4aaa-b0a8-59788b48067a)

For Application Output

<img width="438" alt="Untitled" src="https://github.com/prarthanats/ERA/assets/32382676/dc9f5d46-a714-4a8e-9e4a-cbc34927184f">

<img width="328" alt="Untitled1" src="https://github.com/prarthanats/ERA/assets/32382676/6b7309eb-5564-47fc-be53-4fa0db96a3ae">


###
### Training Log

~~~
	Train Metrics
	Epoch: 0
	Loss: 19.502483367919922
	Class Accuracy: 34.179066%
	No Object Accuracy: 99.917130%
	Object Accuracy: 0.204623%
	100%
	1035/1035 [02:39<00:00, 6.53it/s]
	Train Metrics
	Epoch: 1
	Loss: 12.445760726928711
	Class Accuracy: 36.521210%
	No Object Accuracy: 99.578812%
	Object Accuracy: 8.541598%
	100%
	1035/1035 [02:40<00:00, 6.33it/s]
	Train Metrics
	Epoch: 2
	Loss: 11.295914649963379
	Class Accuracy: 40.674385%
	No Object Accuracy: 96.882538%
	Object Accuracy: 32.231983%
	100%
	1035/1035 [02:40<00:00, 6.61it/s]
	Train Metrics
	Epoch: 3
	Loss: 10.414724349975586
	Class Accuracy: 44.304558%
	No Object Accuracy: 97.067505%
	Object Accuracy: 43.169777%
	100%
	1035/1035 [02:40<00:00, 6.41it/s]
	Train Metrics
	Epoch: 4
	Loss: 9.832324981689453
	Class Accuracy: 40.855396%
	No Object Accuracy: 96.998962%
	Object Accuracy: 40.777958%
	100%
	1035/1035 [02:39<00:00, 7.09it/s]
	Train Metrics
	Epoch: 5
	Loss: 10.241968154907227
	Class Accuracy: 44.617996%
	No Object Accuracy: 97.478615%
	Object Accuracy: 45.524731%
	100%
	1035/1035 [02:39<00:00, 7.34it/s]
	Train Metrics
	Epoch: 6
	Loss: 9.115326881408691
	Class Accuracy: 46.167873%
	No Object Accuracy: 97.849052%
	Object Accuracy: 46.020443%
	100%
	1035/1035 [02:40<00:00, 6.65it/s]
	Train Metrics
	Epoch: 7
	Loss: 8.675603866577148
	Class Accuracy: 48.821156%
	No Object Accuracy: 98.533897%
	Object Accuracy: 44.024731%
	100%
	1035/1035 [02:39<00:00, 8.27it/s]
	Train Metrics
	Epoch: 8
	Loss: 8.37851333618164
	Class Accuracy: 49.815704%
	No Object Accuracy: 96.986244%
	Object Accuracy: 53.893353%

	100%
	1035/1035 [02:42<00:00, 6.41it/s]
	Train Metrics
	Epoch: 9
	Loss: 8.108885765075684
	Class Accuracy: 54.863953%
	No Object Accuracy: 97.961784%
	Object Accuracy: 54.287582%
	100%
	310/310 [00:17<00:00, 18.47it/s]
	Test Metrics
	Class Accuracy: 63.718479%
	No Object Accuracy: 98.858788%
	Object Accuracy: 43.344418%
	100%
	310/310 [11:50<00:00, 1.85s/it]
	MAP:  0.08291830122470856
	100%
	1035/1035 [02:41<00:00, 6.51it/s]
	Train Metrics
	Epoch: 10
	Loss: 7.821064472198486
	Class Accuracy: 56.646358%
	No Object Accuracy: 96.669060%
	Object Accuracy: 63.342415%
	100%
	1035/1035 [02:41<00:00, 6.71it/s]
	Train Metrics
	Epoch: 11
	Loss: 7.634126663208008
	Class Accuracy: 55.954838%
	No Object Accuracy: 97.647102%
	Object Accuracy: 58.734524%
	100%
	1035/1035 [02:40<00:00, 6.82it/s]
	Train Metrics
	Epoch: 12
	Loss: 7.385874271392822
	Class Accuracy: 57.595516%
	No Object Accuracy: 98.339325%
	Object Accuracy: 52.240688%
	100%
	1035/1035 [02:41<00:00, 6.19it/s]
	Train Metrics
	Epoch: 13
	Loss: 7.174068927764893
	Class Accuracy: 61.492401%
	No Object Accuracy: 97.944954%
	Object Accuracy: 60.056599%
	100%
	1035/1035 [02:42<00:00, 6.20it/s]
	Train Metrics
	Epoch: 14
	Loss: 6.914626121520996
	Class Accuracy: 56.621807%
	No Object Accuracy: 97.423264%
	Object Accuracy: 61.451046%
	100%
	1035/1035 [02:42<00:00, 6.55it/s]
	Train Metrics
	Epoch: 15
	Loss: 6.7478108406066895
	Class Accuracy: 63.351494%
	No Object Accuracy: 97.301903%
	Object Accuracy: 65.405815%
	100%
	1035/1035 [02:41<00:00, 8.82it/s]
	Train Metrics
	Epoch: 16
	Loss: 6.573994159698486
	Class Accuracy: 65.826271%
	No Object Accuracy: 97.828804%
	Object Accuracy: 65.816368%
	100%
	1035/1035 [02:41<00:00, 6.34it/s]
	Train Metrics
	Epoch: 17
	Loss: 6.332271099090576
	Class Accuracy: 63.813747%
	No Object Accuracy: 96.895027%
	Object Accuracy: 70.767799%
	100%
	1035/1035 [02:40<00:00, 6.75it/s]
	Train Metrics
	Epoch: 18
	Loss: 6.208128452301025
	Class Accuracy: 68.319359%
	No Object Accuracy: 97.990685%
	Object Accuracy: 65.444275%

	100%
	1035/1035 [02:42<00:00, 6.62it/s]
	Train Metrics
	Epoch: 19
	Loss: 6.101820468902588
	Class Accuracy: 67.979492%
	No Object Accuracy: 97.375778%
	Object Accuracy: 70.518890%
	100%
	310/310 [00:17<00:00, 20.19it/s]
	Test Metrics
	Class Accuracy: 76.328621%
	No Object Accuracy: 98.139610%
	Object Accuracy: 66.558609%
	100%
	310/310 [15:06<00:00, 2.20s/it]
	MAP:  0.19808202981948853
	100%
	1035/1035 [02:42<00:00, 7.43it/s]
	Train Metrics
	Epoch: 20
	Loss: 5.8953938484191895
	Class Accuracy: 68.031242%
	No Object Accuracy: 97.821587%
	Object Accuracy: 64.234901%
	100%
	1035/1035 [02:41<00:00, 7.08it/s]
	Train Metrics
	Epoch: 21
	Loss: 5.769168853759766
	Class Accuracy: 70.401154%
	No Object Accuracy: 97.671303%
	Object Accuracy: 69.599106%
	100%
	1035/1035 [02:42<00:00, 7.17it/s]
	Train Metrics
	Epoch: 22
	Loss: 5.630765438079834
	Class Accuracy: 72.520157%
	No Object Accuracy: 97.842178%
	Object Accuracy: 70.426537%
	100%
	1035/1035 [02:42<00:00, 7.22it/s]
	Train Metrics
	Epoch: 23
	Loss: 5.497457504272461
	Class Accuracy: 74.100502%
	No Object Accuracy: 97.840652%
	Object Accuracy: 70.981270%
	100%
	1035/1035 [02:41<00:00, 8.20it/s]
	Train Metrics
	Epoch: 24
	Loss: 5.3333024978637695
	Class Accuracy: 73.861275%
	No Object Accuracy: 97.801506%
	Object Accuracy: 72.214470%
	100%
	1035/1035 [02:41<00:00, 6.09it/s]
	Train Metrics
	Epoch: 25
	Loss: 5.241975784301758
	Class Accuracy: 73.311028%
	No Object Accuracy: 97.917259%
	Object Accuracy: 72.198837%
	100%
	1035/1035 [02:42<00:00, 6.39it/s]
	Train Metrics
	Epoch: 26
	Loss: 5.088991641998291
	Class Accuracy: 75.655518%
	No Object Accuracy: 97.980888%
	Object Accuracy: 72.182480%
	100%
	1035/1035 [02:42<00:00, 6.90it/s]
	Train Metrics
	Epoch: 27
	Loss: 4.980171203613281
	Class Accuracy: 76.962952%
	No Object Accuracy: 97.774742%
	Object Accuracy: 74.573509%
	100%
	1035/1035 [02:41<00:00, 6.97it/s]
	Train Metrics
	Epoch: 28
	Loss: 4.905660152435303
	Class Accuracy: 77.454941%
	No Object Accuracy: 97.627190%
	Object Accuracy: 76.094421%

	100%
	1035/1035 [02:41<00:00, 6.49it/s]
	Train Metrics
	Epoch: 29
	Loss: 4.744002342224121
	Class Accuracy: 78.818855%
	No Object Accuracy: 97.447975%
	Object Accuracy: 77.391937%
	100%
	310/310 [00:17<00:00, 21.65it/s]
	Test Metrics
	Class Accuracy: 85.397621%
	No Object Accuracy: 98.437660%
	Object Accuracy: 71.978394%
	100%
	310/310 [14:58<00:00, 2.43s/it]
	MAP:  0.36952903866767883
	100%
	1035/1035 [02:42<00:00, 6.69it/s]
	Train Metrics
	Epoch: 30
	Loss: 4.677617073059082
	Class Accuracy: 79.815552%
	No Object Accuracy: 97.882866%
	Object Accuracy: 76.136093%
	100%
	1035/1035 [02:43<00:00, 6.17it/s]
	Train Metrics
	Epoch: 31
	Loss: 4.5202250480651855
	Class Accuracy: 80.811272%
	No Object Accuracy: 97.889565%
	Object Accuracy: 75.987869%
	100%
	1035/1035 [02:43<00:00, 6.74it/s]
	Train Metrics
	Epoch: 32
	Loss: 4.41505241394043
	Class Accuracy: 81.102905%
	No Object Accuracy: 97.848663%
	Object Accuracy: 77.105019%
	100%
	1035/1035 [02:44<00:00, 6.72it/s]
	Train Metrics
	Epoch: 33
	Loss: 4.3231282234191895
	Class Accuracy: 81.923737%
	No Object Accuracy: 97.869797%
	Object Accuracy: 77.460396%
	100%
	1035/1035 [02:43<00:00, 6.64it/s]
	Train Metrics
	Epoch: 34
	Loss: 4.202864170074463
	Class Accuracy: 82.886543%
	No Object Accuracy: 97.863792%
	Object Accuracy: 78.154495%
	100%
	1035/1035 [02:43<00:00, 6.52it/s]
	Train Metrics
	Epoch: 35
	Loss: 4.086621284484863
	Class Accuracy: 82.883827%
	No Object Accuracy: 97.916573%
	Object Accuracy: 77.908768%
	100%
	1035/1035 [02:42<00:00, 6.12it/s]
	Train Metrics
	Epoch: 36
	Loss: 3.994511842727661
	Class Accuracy: 84.447655%
	No Object Accuracy: 97.978493%
	Object Accuracy: 79.005524%
	100%
	1035/1035 [02:43<00:00, 6.42it/s]
	Train Metrics
	Epoch: 37
	Loss: 3.8890902996063232
	Class Accuracy: 84.914017%
	No Object Accuracy: 97.986221%
	Object Accuracy: 78.855965%
	100%
	1035/1035 [02:42<00:00, 6.28it/s]
	Train Metrics
	Epoch: 38
	Loss: 3.815894365310669
	Class Accuracy: 85.592529%
	No Object Accuracy: 98.074226%
	Object Accuracy: 78.947586%

	100%
	1035/1035 [02:42<00:00, 6.01it/s]
	Train Metrics
	Epoch: 39
	Loss: 3.7511277198791504
	Class Accuracy: 85.762337%
	No Object Accuracy: 98.037003%
	Object Accuracy: 79.472855%
	100%
	310/310 [00:17<00:00, 19.37it/s]
	Test Metrics
	Class Accuracy: 89.573288%
	No Object Accuracy: 98.923180%
	Object Accuracy: 72.610138%
	100%
	310/310 [10:22<00:00, 1.64s/it]
	MAP:  0.4797763228416443
~~~
