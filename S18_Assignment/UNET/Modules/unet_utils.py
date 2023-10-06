# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:17:47 2023
@author: prarthana.ts
"""

import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# View an example of image and corresponding mask 
def sample_images(path1, path2, img, mask, show_images = 1):
    for i in range(show_images):
        img_view  = imageio.imread(path1 + img[i])
        mask_view = imageio.imread(path2 + mask[i])
        print(img_view.shape)
        print(mask_view.shape)
        fig, arr = plt.subplots(1, 2, figsize=(15, 15))
        arr[0].imshow(img_view)
        arr[0].set_title('Image '+ str(i))
        arr[1].imshow(mask_view)
        arr[1].set_title('Masked Image '+ str(i))


# Visualize the processed output
def show_processed_image(X, y, image_index = 0):
    fig, arr = plt.subplots(1, 2, figsize=(15, 15))
    arr[0].imshow(X[image_index])
    arr[0].set_title('Processed Image')
    arr[1].imshow(y[image_index,:,:,0])
    arr[1].set_title('Processed Masked Image ')


# Bias Variance Check
# High Bias is a characteristic of an underfitted model and we would observe low accuracies for both train and validation set
# High Variance is a characterisitic of an overfitted model and we would observe high accuracy for train set and low for validation set
# To check for bias and variance plot the graphs for accuracy 
def model_metrics(results):
    fig, axis = plt.subplots(1, 2, figsize=(20, 5))
    axis[0].plot(results.history["loss"], color='r', label = 'train loss')
    axis[0].plot(results.history["val_loss"], color='b', label = 'dev loss')
    axis[0].set_title('Loss Comparison')
    axis[0].legend()
    axis[1].plot(results.history["accuracy"], color='r', label = 'train accuracy')
    axis[1].plot(results.history["val_accuracy"], color='b', label = 'dev accuracy')
    axis[1].set_title('Accuracy Comparison')
    axis[1].legend()

# Results of Validation Dataset
def output_visualize(X_valid, unet, y_valid, index):
    img = X_valid[index]
    img = img[np.newaxis, ...]
    pred_y = unet.predict(img)
    pred_mask = tf.argmax(pred_y[0], axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    fig, arr = plt.subplots(1, 3, figsize=(15, 15))
    arr[0].imshow(X_valid[index])
    arr[0].set_title('Processed Image')
    arr[1].imshow(y_valid[index,:,:,0])
    arr[1].set_title('Actual Masked Image ')
    arr[2].imshow(pred_mask[:,:,0])
    arr[2].set_title('Predicted Masked Image ')