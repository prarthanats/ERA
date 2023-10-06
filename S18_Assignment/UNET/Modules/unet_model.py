# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:17:05 2023
@author: prarthana.ts
"""

# for bulding and running deep learning model
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy  
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Conv2DTranspose, UpSampling2D, concatenate
from tensorflow.keras.metrics import MeanIoU

import tensorflow as tf
import tensorflow.keras.backend as K

def dice_loss(y_true, y_pred, n_classes=3, smooth=0.001):
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_classes)
    
    y_pred = tf.cast(tf.reshape(y_pred, [-1, n_classes]), tf.float32)
    y_true = tf.cast(tf.reshape(y_true, [-1, n_classes]), tf.float32)
    
    intersection = 2 * tf.reduce_sum(y_true * y_pred, axis=0)
    union = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0)
    
    dice = (intersection + smooth) / (union + smooth)
    
    return 1 - K.mean(dice)


def binary_cross_entropy_loss(y_true, y_pred, n_classes=3):
    # Step 1: Convert prediction to softmax probabilities
    y_pred = tf.nn.softmax(y_pred, axis=-1)

    # Step 2: Convert target to one-hot format
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_classes)

    # Flatten both prediction and target tensors
    y_pred = tf.cast(tf.reshape(y_pred, [-1, n_classes]), tf.float32)
    y_true = tf.cast(tf.reshape(y_true, [-1, n_classes]), tf.float32)

    # Compute binary cross-entropy loss
    loss = categorical_crossentropy  (y_true, y_pred)
    return loss

def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, use_max_pooling=True, use_strided_conv=False, use_upsampling=False):
    # Convolutional layers
    conv1 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv2 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    
    # Batch normalization
    conv2 = BatchNormalization()(conv2, training=False)

    # Dropout layer (if dropout_prob > 0)
    if dropout_prob > 0:
        conv2 = Dropout(dropout_prob)(conv2)

    # Pooling or convolution based on the specified options
    if use_max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv2)
    elif use_strided_conv:
        next_layer = Conv2D(n_filters, 3, activation='relu', padding='same', strides=2)(conv2)
    elif use_upsampling:
        next_layer = UpSampling2D()(conv2)
    else:
        next_layer = conv2

    skip_connection = conv2
    
    return next_layer, skip_connection

def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32, use_transpose_conv=True, use_upsampling=False):
    # Upsampling or transpose convolution based on the specified options
    if use_transpose_conv:
        up = Conv2DTranspose(n_filters, (3, 3), strides=2, activation='relu', padding='same')(prev_layer_input)
    elif use_upsampling:
        up = UpSampling2D()(prev_layer_input)
        up = Conv2D(n_filters, 3, activation='relu', padding='same')(up)
    else:
        up = prev_layer_input

    # Ensure the dimensions match by cropping or padding the skip_layer_input
    target_shape = up.shape[1:3]  # Target spatial dimensions
    skip_layer_input = tf.image.resize(skip_layer_input, target_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Concatenate the upsampled features with skip_layer_input
    merge = concatenate([up, skip_layer_input], axis=3)
    
    # Additional convolutional layers
    conv1 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
    conv2 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    
    return conv2  # Returning conv2 for consistency with the EncoderMiniBlock


def UNetModel(input_size=(128, 128, 3), n_filters=32, n_classes=3, use_max_pooling=True, use_transpose_conv=True, use_strided_conv=False, use_upsampling=False, use_dice_loss=False, use_bce=False):
    inputs = Input(input_size)
    
    cblock1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, use_max_pooling=use_max_pooling, use_strided_conv=use_strided_conv, use_upsampling=use_upsampling)
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters*2, dropout_prob=0, use_max_pooling=use_max_pooling, use_strided_conv=use_strided_conv, use_upsampling=use_upsampling)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4, dropout_prob=0, use_max_pooling=use_max_pooling, use_strided_conv=use_strided_conv, use_upsampling=use_upsampling)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8, dropout_prob=0.3, use_max_pooling=use_max_pooling, use_strided_conv=use_strided_conv, use_upsampling=use_upsampling)
    cblock5 = EncoderMiniBlock(cblock4[0], n_filters*16, dropout_prob=0.3, use_max_pooling=False, use_strided_conv=False, use_upsampling=False) 

    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1],  n_filters * 8, use_transpose_conv=use_transpose_conv, use_upsampling=use_upsampling)
    ublock7 = DecoderMiniBlock(ublock6, cblock3[1],  n_filters * 4, use_transpose_conv=use_transpose_conv, use_upsampling=use_upsampling)
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1],  n_filters * 2, use_transpose_conv=use_transpose_conv, use_upsampling=use_upsampling)
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1],  n_filters, use_transpose_conv=use_transpose_conv, use_upsampling=use_upsampling)

    conv9 = Conv2D(n_filters,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    if use_dice_loss:
        loss = dice_loss
    elif use_bce:
        loss = binary_cross_entropy_loss 
    
    
    print("Model Summary:")
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=['accuracy'])
    return model