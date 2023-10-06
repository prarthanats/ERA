# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:10:45 2023
@author: prarthana.ts
"""

# for data load
import os
import numpy as np # for using np arrays
import urllib.request
import tarfile

# for reading and processing images
from PIL import Image


# Function to download and extract the dataset
def download_and_extract(url, download_dir, filename, extract_dir):
    # Download the file
    urllib.request.urlretrieve(url, os.path.join(download_dir, filename))
    
    # Extract the tar.gz file
    with tarfile.open(os.path.join(download_dir, filename), "r:gz") as tar:
        tar.extractall(extract_dir)


# Helper Functions for Data Processing
# Load Data
# the masked images are stored as png, unmasked (original) as jpg
# the names of these 2 are same so for getting the right sample we can just sort the 2 lists
def LoadData(path1, path2):
    """
    Looks for relevant filenames in the shared path
    Returns 2 lists for original and masked files respectively
    """
    # Read the images folder like a list
    image_dataset = os.listdir(path1)

    # Filter out hidden files (those starting with a period)
    image_dataset = [file for file in image_dataset if not file.startswith('.')]
    image_dataset = [file for file in image_dataset if not file.endswith('.mat')]

    mask_dataset = os.listdir(path2)

    # Filter out hidden files in the annotations folder as well
    mask_dataset = [file for file in mask_dataset if not file.startswith('.')]
    mask_dataset = [file for file in mask_dataset if not file.endswith('.mat')]

    # Make a list for images and masks filenames
    orig_img = []
    mask_img = []

    for file in image_dataset:
        orig_img.append(file)

    for file in mask_dataset:
        mask_img.append(file)

    # Sort the lists to get both of them in the same order (the dataset has exactly the same name for images and corresponding masks)
    orig_img.sort()
    mask_img.sort()

    return orig_img, mask_img



# Pre-Process Data
def PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2):
    """
    Processes the images and mask present in the shared list and path
    Returns a NumPy dataset with images as 3-D arrays of desired size
    Please note the masks in this dataset have only one channel
    """
    # Pull the relevant dimensions for image and mask
    m = len(img)                     # number of images
    i_h,i_w,i_c = target_shape_img   # pull height, width, and channels of image
    m_h,m_w,m_c = target_shape_mask  # pull height, width, and channels of mask
    
    # Define X and Y as number of images along with shape of one image
    X = np.zeros((m,i_h,i_w,i_c), dtype=np.float32)
    y = np.zeros((m,m_h,m_w,m_c), dtype=np.int32)
    total_images = len(img)
    count=0
    
    # Resize images and masks
    for file in img:
        print(f"\rProcessing image {count + 1} out of {total_images} images", end='', flush=True)
        count = count + 1
        # convert image into an array of desired shape (3 channels)
        index = img.index(file)
        path = os.path.join(path1, file)
        single_img = Image.open(path).convert('RGB')
        single_img = single_img.resize((i_h,i_w))
        single_img = np.reshape(single_img,(i_h,i_w,i_c)) 
        single_img = single_img/256.
        X[index] = single_img
        
        # convert mask into an array of desired shape (1 channel)
        single_mask_ind = mask[index]
        path = os.path.join(path2, single_mask_ind)
        single_mask = Image.open(path)
        single_mask = single_mask.resize((m_h, m_w))
        single_mask = np.reshape(single_mask,(m_h,m_w,m_c)) 
        single_mask = single_mask - 1 # to ensure classes #s start from 0
        y[index] = single_mask
    print()
    return X, y