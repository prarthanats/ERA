# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:17:23 2023
@author: prarthana.ts
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

def unet_train(unet, X, y):
    # Split Train and Test Set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=123)
    # Run the model in a mini-batch fashion and compute the progress for each epoch
    results = unet.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_valid, y_valid))
    return results, unet, X_train, X_valid, y_train, y_valid