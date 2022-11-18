
# Modules
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import tensorflow as tf


def distance_layer(vectors):
	featsA, featsB  = vectors[0], vectors[1]
	sum_squared     = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
	return K.sqrt(sum_squared)

def dense_block(inputs, units:int=1, dropout:float=0, name:str=''):
    dense         = layers.Dense(units=units, activation='relu', use_bias=False, kernel_initializer='he_normal', name=f'{name}_dense')(inputs)
    normalisation = layers.BatchNormalization(name=f'{name}_normalisation')(dense)
    outputs       = layers.Dropout(rate=dropout, name=f'{name}_dropout')(normalisation)
    return outputs

def convolution_block(inputs, filters:int, dropout:float=0, name:str=''):
    tensor = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_convolution1')(inputs)
    tensor = layers.Activation('relu', name=f'{name}_activation1')(tensor)
    tensor = layers.BatchNormalization(name=f'{name}_normalisation1')(tensor)
    tensor = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_convolution2')(tensor)
    tensor = layers.Activation('relu', name=f'{name}_activation2')(tensor)
    tensor = layers.BatchNormalization(name=f'{name}_normalisation2')(tensor)
    tensor = layers.MaxPool2D(pool_size=(2, 2), name=f'{name}_pooling')(tensor)
    tensor = layers.SpatialDropout2D(rate=dropout, name=f'{name}_dropout')(tensor)

    return tensor


def encoder_block_shared(shape:tuple, units:int, filters:int=1, dropout=0):
    inputs  = layers.Input(shape=shape, name='inputs')
    tensor  = convolution_block(inputs, filters=filters*1, dropout=dropout, name='block1')
    tensor  = convolution_block(tensor, filters=filters*2, dropout=dropout*0.8, name='block2')
    tensor  = convolution_block(tensor, filters=filters*3, dropout=dropout*0.6, name='block3')
    tensor  = convolution_block(tensor, filters=filters*4, dropout=dropout*0.4,  name='block4')
    tensor  = convolution_block(tensor, filters=filters*5, dropout=dropout*0.2, name='block5')
    tensor  = layers.Flatten(name='encoder_flatten')(tensor)
    tensor  = dense_block(tensor, units=units, dropout=dropout*0.2,  name='dense_block1')
    tensor  = dense_block(tensor, units=units, dropout=dropout*0.2, name='dense_block2')
    tensor  = dense_block(tensor, units=units, dropout=dropout*0.2,  name='dense_block3')
    tensor  = dense_block(tensor, units=units, dropout=dropout*0.2, name='dense_block4')
    outputs = dense_block(tensor, units=units,   name='dense_block5')

    encoder = models.Model(inputs=inputs, outputs=outputs, name='encoder')
    return encoder

def siamese_convolutional_network(shape:tuple, args:dict):
    images1         = layers.Input(shape=shape, name='images_t0')
    images2         = layers.Input(shape=shape, name='images_tt')
    encoder_block   = encoder_block_shared(shape=shape, **args)
    encode1         = encoder_block(images1)
    encode2         = encoder_block(images2)
    distance        = layers.Lambda(distance_layer)([encode1, encode2])
    outputs         = layers.Dense(units=1, activation='sigmoid', name='outputs')(distance)
    model           = models.Model(inputs=[images1, images2], outputs=outputs, name='siamese_convolutional_network')
    return model