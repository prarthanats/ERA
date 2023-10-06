# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:11:25 2023
@author: prarthana.ts
"""

from ast import main
import numpy as np # for using np arrays
from unet_dataset import *
from unet_model import *
from unet_utils import *
from unet_train import *

if __name__ == "__main__":

    # Define the URL and file names
    dataset_url = "http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    dataset_filename = "images.tar.gz"
    annotations_filename = "annotations.tar.gz"

    # Define the download and extraction directory
    download_dir = "/content/"
    extracted_dir = "/content/"
    # Download and extract the dataset and annotations
    download_and_extract(dataset_url, download_dir, dataset_filename, extracted_dir)
    download_and_extract(annotations_url, download_dir, annotations_filename, extracted_dir)

    # Define the paths to the images and masks directories
    images_dir = os.path.join(extracted_dir, "images/images")
    masks_dir = os.path.join(extracted_dir, "annotations/annotations/trimaps")


    # Load and View Data
    """ Load Train Set and view some examples """
    # Call the apt function
    path1 = images_dir
    path2 = masks_dir
    img, mask = LoadData (path1, path2)

    show_sample_images(path1, path2, img, mask, show_images = 1)

    # Process Data
    # Define the desired shape
    target_shape_img = [128, 128, 3]
    target_shape_mask = [128, 128, 1]

    # Process data using apt helper function
    X, y = PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2)

    # QC the shape of output and classes in output dataset 
    print("X Shape:", X.shape)
    print("Y shape:", y.shape)
    # There are 3 classes : background, pet, outline 

    show_processed_image(X, y, image_index = 0)
    
    unet = UNetCompiled(input_size=(128, 128, 3), n_filters=32, n_classes=3, use_max_pooling=False, 
                        use_transpose_conv=False, use_strided_conv=True, use_upsampling=True, use_dice_loss=True, use_bce=False)
    
    # Train the model
    results, unet, X_train, X_valid, y_train, y_valid = unet_train(unet, X, y)

    # Evaluate Model Results
    model_metrics(results)

    # RESULTS
    # The train loss is consistently decreasing showing that Adam is able to optimize the model and find the minima
    # The accuracy of train and validation is ~90% which is high enough, so low bias
    # and the %s aren't that far apart, hence low variance

    # View Predicted Segmentations
    unet.evaluate(X_valid, y_valid)

    # Add any index to contrast the predicted mask with actual mask
    index = 700
    VisualizeResults(X_valid, unet, y_valid, index)