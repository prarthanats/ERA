# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 19:53:33 2023
@author: prarthana.ts
"""

import matplotlib.pyplot as plt
import numpy as np

def class_image(train,classes): 
    class_dict = {}

    for i, classs in enumerate(classes):
      images_list = [index for index, (image, label) in enumerate(train) if label == i]
      class_dict[classs] = np.random.choice(images_list, 5, replace = False)
  
    fig, axes = plt.subplots(ncols=6, nrows=10, figsize=(5, 10))
    index = 0
    for i, classlabel in enumerate(classes):
        for j in range(1, 6):
          axes[i][0].text(x = 0, y = 0.3, s = classlabel, rotation = 0, va = "center_baseline")
          axes[i,j].imshow(np.transpose(train[class_dict[classlabel][j-1]][0].numpy(), (1,2,0)))
          axes[i,j].get_xaxis().set_visible(False)
          axes[i,j].get_yaxis().set_visible(False)
          axes[i][0].get_xaxis().set_visible(False)
          axes[i][0].get_yaxis().set_visible(False)
          axes[i][0].spines['top'].set_visible(False)
          axes[i][0].spines['right'].set_visible(False)
          axes[i][0].spines['bottom'].set_visible(False)
          axes[i][0].spines['left'].set_visible(False)
          index += 1
    plt.axis('off')
    plt.show()
    
def show_misclassified_img(misclassified_images, misclassified_labels, misclassified_predictions, classes):
    fig = plt.figure(figsize=(20, 4))
    for i in range(len(misclassified_images)):
        ax = fig.add_subplot(2, 5, i + 1)
        image = misclassified_images[i].cpu().numpy().transpose(1, 2, 0)
        label = misclassified_labels[i].cpu().numpy().item()  # Convert to integer
        prediction = misclassified_predictions[i].cpu().numpy().item()  # Convert to integer
        # Normalize image data
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        ax.imshow(image)
        ax.set_title(f'Expected: {classes[label]}\nPredicted: {classes[prediction]}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def show_accuracy_loss(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")