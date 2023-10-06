
# Training UNET - Oxford Pet Dataset

U-Net with 4 different strategies.
~~~
  1. MP+Tr+CE
  2. MP+Tr+Dice Loss
  3. StrConv+Tr+CE
  4. StrConv+Ups+Dice Loss
~~~

## Introduction to U-Net

U-Net, introduced in 2015 by Olaf Ronneberger and his team, was initially designed for biomedical image analysis. It excelled in the ISBI challenge by surpassing the sliding window technique in performance while using fewer images and data augmentation.

The sliding window approach is effective for localization tasks, creating local patches and individual class labels for each pixel. However, it has two significant drawbacks: it generates redundancy due to overlapping patches and is computationally slow during training, consuming considerable time and resources. These limitations made it impractical for many applications. U-Net addresses these shortcomings.

U-Net's architecture resembles a "U" shape, consisting of convolutional layers and two networks: an encoder followed by a decoder. U-Net effectively handles both aspects of segmentation: "what" and "where." It's a powerful tool for solving segmentation problems in various domains.

![image](https://github.com/prarthanats/ERA/assets/32382676/229b95ae-1a28-4cfc-9787-535c0695035e)

Here's a detailed description of the UNet architecture:

Encoder-Decoder Structure: UNet is characterized by its U-shaped architecture, which consists of two main parts: the encoder and the decoder. This design allows for the extraction of features through the encoder and then the precise localization of objects through the decoder.

Contracting Path (Encoder): The encoder component of UNet is designed to capture context and features from the input image. It typically comprises multiple convolutional layers, often arranged in a downsampling fashion. As you move deeper into the encoder, the spatial dimensions of the feature maps decrease, but the number of channels (feature maps) typically increases. This helps in abstracting and capturing high-level features.

Expansive Path (Decoder): The decoder, on the other hand, is responsible for gradually increasing the spatial resolution of the feature maps while maintaining a rich representation of features. It uses transposed convolutional layers or upsampling operations to achieve this. Skip connections are a key feature of UNet that connect corresponding layers from the encoder to the decoder. These skip connections help in preserving fine-grained details and improve the segmentation results.

Skip Connections: The skip connections bridge the gap between the encoder and decoder by concatenating or adding feature maps from the encoder to the decoder at each corresponding level. This allows the model to combine both low-level and high-level features, which is crucial for accurate segmentation. Skip connections enable UNet to localize objects precisely.

Final Layer: The decoder's output layer typically consists of a single convolutional layer with a sigmoid activation function. This layer generates a probability map where each pixel represents the likelihood of belonging to the target object class. Post-processing techniques such as thresholding or connected component analysis are often applied to produce the final binary segmentation mask.

Loss Function: The training of a UNet model involves minimizing a suitable loss function, such as binary cross-entropy or dice coefficient loss, which quantifies the dissimilarity between the predicted segmentation mask and the ground truth mask.

## Modification 1 - Max Pooling(Contracting) + Transpose Convolutions(Expanding) + Cross Entropy Loss

Max Pooling (Contracting): Max Pooling is a downsampling operation that reduces the spatial dimensions of the feature maps. It achieves this by selecting the maximum value in a small window, effectively compressing the information. In the U-Net's contracting path, Max Pooling helps capture and abstract high-level features while reducing the computational burden by shrinking the feature map size. This allows the model to understand the "what" (semantic information) in the input image.

Transpose Convolutions (Expanding): Transpose Convolutions are used in the expanding path of U-Net to increase the spatial dimensions of feature maps. The expanding path helps the model understand the "where" (pixel-level localization) of objects or structures in the image.

Cross Entropy Loss:Cross Entropy Loss is a common choice for segmentation tasks, quantifying the difference between predicted and actual pixel-wise class probabilities. U-Net trains to accurately classify each pixel, distinguishing between object and background classes.

![image](https://github.com/prarthanats/ERA/assets/32382676/f6989583-5f80-4742-9a76-79cc32c670a7)

![image](https://github.com/prarthanats/ERA/assets/32382676/c16e0e44-1f76-4609-a53a-19e7ae33dbbc)

## Modification 2 - Max Pooling(Contracting) + Transpose Convolutions(Expanding) + Dice Loss

Dice Loss: Dice Loss is a similarity metric commonly used for image segmentation tasks. It measures the overlap between predicted and ground truth segmentation masks, rewarding accurate pixel-level predictions. Incorporating Dice Loss into the training process encourages the model to produce more precise and spatially coherent segmentations.

By combining Max Pooling for semantic understanding, Transpose Convolutions for fine-grained detail recovery, and Dice Loss for improved segmentation accuracy, U-Net becomes a powerful architecture for tasks that require both high-level semantic recognition and precise localization, such as medical image segmentation. This modification helps strike a balance between capturing "what" is in the image and "where" it is located while optimizing segmentation performance.

![image](https://github.com/prarthanats/ERA/assets/32382676/80c0f41b-5b13-4228-a901-34e06f9b93e7)

![image](https://github.com/prarthanats/ERA/assets/32382676/9547cf06-5ab1-4b98-8d0a-d5005443d51c)


