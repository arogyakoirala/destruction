#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Initlaises models
@author: Clement Gorin and Arogya
@contact: gorinclem@gmail.com
@version: 2022.06.01
'''

# Modules
# from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import tensorflow as tf


def dense_block(inputs, units:int=1, dropout:float=0, name:str=''):
    dense         = layers.Dense(units=units, activation='relu', use_bias=False, kernel_initializer='he_normal', name=f'{name}_dense')(inputs)
    normalisation = layers.BatchNormalization(name=f'{name}_normalisation')(dense)
    outputs       = layers.Dropout(rate=dropout, name=f'{name}_dropout')(normalisation)
    return outputs

def convolution_block(inputs, filters:int, dropout:float=0, name:str=''):
    convolution   = layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_convolution')(inputs)
    pooling       = layers.MaxPool2D(pool_size=(2, 2), name=f'{name}_pooling')(convolution)
    normalisation = layers.BatchNormalization(name=f'{name}_normalisation')(pooling)
    outputs       = layers.SpatialDropout2D(rate=dropout, name=f'{name}_dropout')(normalisation)
    return outputs

def convolutional_network(shape:tuple, filters:int, units:int, dropout:float):
    # Input layer
    inputs = layers.Input(shape=shape, name='inputs')
    # Hidden convolutional layers
    tensor = convolution_block(inputs, filters=filters*1, dropout=dropout, name='conv_block1')
    tensor = convolution_block(tensor, filters=filters*2, dropout=dropout, name='conv_block2')
    tensor = convolution_block(tensor, filters=filters*3, dropout=dropout, name='conv_block3')
    tensor = convolution_block(tensor, filters=filters*4, dropout=dropout, name='conv_block4')
    tensor = convolution_block(tensor, filters=filters*5, dropout=dropout, name='conv_block5')
    # Hidden dense layers
    tensor = layers.Flatten(name='flatten')(tensor)
    tensor = dense_block(tensor, units=units, dropout=dropout, name='dense_block1')
    tensor = dense_block(tensor, units=units, dropout=dropout, name='dense_block2')
    # Output layer
    outputs = layers.Dense(units=1, activation='sigmoid', name='outputs')(tensor)
    # Model
    model   = models.Model(inputs=inputs, outputs=outputs, name='convolutional_network')
    return model
