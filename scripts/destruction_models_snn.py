#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Initialises models with chained convolutions
@author: Clement Gorin and Arogya
@contact: gorinclem@gmail.com
@version: 2022.11.06
'''

# Modules
from tensorflow.keras import backend, layers, models
from tensorflow import math

def dense_block(inputs, units:int=1, dropout:float=0, name:str=''):
    tensor = layers.Dense(units=units, use_bias=False, kernel_initializer='he_normal', name=f'{name}_dense')(inputs)
    tensor = layers.Activation('relu', name=f'{name}_activation')(tensor)
    tensor = layers.BatchNormalization(name=f'{name}_normalisation')(tensor)
    tensor = layers.Dropout(rate=dropout, name=f'{name}_dropout')(tensor)
    return tensor 

def convolution_block(inputs, filters:int, dropout:float, name:str):
    tensor = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_convolution1')(inputs)
    tensor = layers.Activation('relu', name=f'{name}_activation1')(tensor)
    tensor = layers.BatchNormalization(name=f'{name}_normalisation1')(tensor)
    tensor = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_convolution2')(tensor)
    tensor = layers.Activation('relu', name=f'{name}_activation2')(tensor)
    tensor = layers.BatchNormalization(name=f'{name}_normalisation2')(tensor)
    tensor = layers.MaxPool2D(pool_size=(2, 2), name=f'{name}_pooling')(tensor)
    tensor = layers.SpatialDropout2D(rate=dropout, name=f'{name}_dropout')(tensor)
    return tensor

def encoder_block_separated(inputs, filters:int=1, dropout:float=0, name:str=''):
    tensor = convolution_block(inputs, filters=filters*1, dropout=dropout, name=f'{name}_block1')
    tensor = convolution_block(tensor, filters=filters*2, dropout=dropout, name=f'{name}_block2')
    tensor = convolution_block(tensor, filters=filters*3, dropout=dropout, name=f'{name}_block3')
    tensor = convolution_block(tensor, filters=filters*4, dropout=dropout, name=f'{name}_block4')
    tensor = convolution_block(tensor, filters=filters*5, dropout=dropout, name=f'{name}_block5')
    tensor = layers.GlobalAveragePooling2D(name=f'{name}_global_pooling')(tensor)
    return tensor

def encoder_block_shared(shape:tuple, filters:int=1, dropout=0):
    inputs  = layers.Input(shape=shape, name='inputs')
    tensor  = convolution_block(inputs, filters=filters*1, dropout=dropout, name='block1')
    tensor  = convolution_block(tensor, filters=filters*2, dropout=dropout, name='block2')
    tensor  = convolution_block(tensor, filters=filters*3, dropout=dropout, name='block3')
    tensor  = convolution_block(tensor, filters=filters*4, dropout=dropout, name='block4')
    tensor  = convolution_block(tensor, filters=filters*5, dropout=dropout, name='block5')
    outputs = layers.GlobalAveragePooling2D(name='global_pooling')(tensor)
    encoder = models.Model(inputs=inputs, outputs=outputs, name='encoder')
    return encoder

def distance_layer(inputs):
    input0, input1 = inputs
    distances = math.reduce_sum(math.square(input0 - input1), axis=1, keepdims=True)
    distances = math.sqrt(math.maximum(distances, backend.epsilon()))
    return distances

def siamese_convolutional_network(shape:tuple, args_encode:dict, args_dense:dict):
    encoder = encoder_block_shared(shape=shape, **args_encode)
    images1 = layers.Input(shape=shape, name='images_t0')
    images2 = layers.Input(shape=shape, name='images_tt')
    encode1 = encoder(images1)
    encode2 = encoder(images2)
    dist    = distance_layer([encode1, encode2])
    dense   = dense_block(dist,  **args_dense, name='dense_block1')
    dense   = dense_block(dense, **args_dense, name='dense_block2')
    dense   = dense_block(dense, **args_dense, name='dense_block3')
    outputs = layers.Dense(units=1, activation='sigmoid', name='outputs')(dense)
    model   = models.Model(inputs=[images1, images2], outputs=outputs, name='siamese_convolutional_network')
    return model

def double_convolutional_network(shape:tuple, args_encode:dict, args_dense:dict):
    images1 = layers.Input(shape=shape, name='images_t0')
    images2 = layers.Input(shape=shape, name='images_tt')
    encode1 = encoder_block_separated(images1, **args_encode, name='encoder1')
    encode2 = encoder_block_separated(images2, **args_encode, name='encoder2')
    concat  = layers.Concatenate(name='concatenate')([encode1, encode2])
    dense   = dense_block(concat, **args_dense, name='dense_block1')
    dense   = dense_block(dense,  **args_dense, name='dense_block2')
    dense   = dense_block(dense,  **args_dense, name='dense_block3')
    outputs = layers.Dense(units=1, activation='sigmoid', name='outputs')(dense)
    model   = models.Model(inputs=[images1, images2], outputs=outputs, name='double_convolutional_network')
    return model


