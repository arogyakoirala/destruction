#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Initlaises models
@author: Clement Gorin and Arogya
@contact: gorinclem@gmail.com
@version: 2022.06.01
'''

# Modules
from tensorflow.keras.utils import Sequence



def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	featsA, featsB = vectors[0], vectors[1]
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(sumSquared)

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

def encoder_block_shared(shape:tuple, filters:int=1, dropout=0):
    inputs  = layers.Input(shape=shape, name='inputs')
    tensor  = convolution_block(inputs, filters=filters*1, dropout=dropout, name='block1')
    tensor  = convolution_block(tensor, filters=filters*2, dropout=dropout, name='block2')
    tensor  = convolution_block(tensor, filters=filters*3, dropout=dropout, name='block3')
    tensor  = convolution_block(tensor, filters=filters*4, dropout=dropout, name='block4')
    tensor  = convolution_block(tensor, filters=filters*5, dropout=dropout, name='block5')
    outputs = layers.Flatten(name='encoder_flatten')(tensor)
    encoder = models.Model(inputs=inputs, outputs=outputs, name='encoder')
    return encoder



def siamese_convolutional_network(shape:tuple, args_encode:dict, args_dense:dict):
    # Input layers
    images1 = layers.Input(shape=shape, name='images_t0')
    images2 = layers.Input(shape=shape, name='images_tt')
    # Hidden convolutional layers (shared parameters)
    encoder_block = encoder_block_shared(shape=shape, **args_encode)
    encode1 = encoder_block(images1)
    encode2 = encoder_block(images2)
    # Hidden dense layers
    concat  = layers.Concatenate(name='concatenate')(inputs=[encode1, encode2])

    units, dropout = args_dense['units'], args_dense['dropout']
    dense   = dense_block(concat, units = units, dropout=dropout, name='dense_block1')
    dense   = dense_block(dense, units = units, dropout=dropout/2, name='dense_block2')
    dense   = dense_block(dense, units = units, dropout=dropout/4, name='dense_block3')
    # Output layer
    outputs = layers.Dense(units=1, activation='sigmoid', name='outputs')(dense)
    # Model
    model   = models.Model(inputs=[images1, images2], outputs=outputs, name='siamese_convolutional_network')
    return model

def siamese_block_shared(shape:tuple, units:int, filters:int=1, dropout=0):
    inputs  = layers.Input(shape=shape, name='inputs')
    tensor  = convolution_block(inputs, filters=filters*1, dropout=dropout, name='block1')
    tensor  = convolution_block(tensor, filters=filters*2, dropout=dropout, name='block2')
    tensor  = convolution_block(tensor, filters=filters*3, dropout=dropout, name='block3')
    tensor  = convolution_block(tensor, filters=filters*4, dropout=dropout,  name='block4')
    tensor  = convolution_block(tensor, filters=filters*5, dropout=dropout, name='block5')
    tensor  = layers.Flatten(name='encoder_flatten')(tensor)
    tensor  = dense_block(tensor, units=units, dropout=dropout,  name='dense_block1')
    tensor  = dense_block(tensor, units=units, dropout=dropout, name='dense_block2')
    tensor  = dense_block(tensor, units=units, dropout=dropout,  name='dense_block3')
    tensor = dense_block(tensor, units=units, dropout=dropout, name='dense_block4')
    outputs = dense_block(tensor, units=units,   name='dense_block5')

    encoder = models.Model(inputs=inputs, outputs=outputs, name='encoder')
    return encoder

def siamese_convolutional_network_dist(shape:tuple, args:dict):
    # Input layers
    images1 = layers.Input(shape=shape, name='images_t0')
    images2 = layers.Input(shape=shape, name='images_tt')
    # Hidden convolutional layers (shared parameters)
    encoder_block = siamese_block_shared(shape=shape, **args)
    encode1 = encoder_block(images1)
    encode2 = encoder_block(images2)
    # Hidden dense layers
    distance = layers.Lambda(euclidean_distance)([encode1, encode2])
    outputs = layers.Dense(units=1, activation='sigmoid', name='outputs')(distance)

    # Output layer
    # Model
    model   = models.Model(inputs=[images1, images2], outputs=outputs, name='siamese_convolutional_network_dist')
    return model

 





class SiameseGenerator(Sequence):
    def __init__(self, images, labels, batch_size=32):
        self.images_t0 = images[0]
        self.images_tt = images[1]
        self.labels = labels
        self.batch_size = batch_size
        
        self.tuple_pairs = utils.make_tuple_pair(self.images_t0.shape[0], int(self.batch_size/4))
        np.random.shuffle(self.tuple_pairs)
   
    def __len__(self):
        return len(self.images_t0)//self.batch_size    
    
    def __getitem__(self, index):
        return self.get_sub_batch(index)

    def get_sub_batch(self,index):
        pos = index*4
        
        t, h, w, d = self.images_t0.shape
        X_t0_main = np.empty((0, h, w,d))
        X_tt_main = np.empty((0, h, w,d))
        y_main = np.empty(0)
#         X_t0_main = np.empty(0, h, w,d)
        for i in range(0,4):
            index_range = self.tuple_pairs[pos+i]
            X_t0 = self.images_t0[index_range[0]:index_range[1]]
            X_tt = self.images_tt[index_range[0]:index_range[1]]
            y = self.labels[index_range[0]:index_range[1]]
            
            X_t0_main=np.append(X_t0_main, X_t0, axis=0)
            X_tt_main=np.append(X_tt_main, X_tt, axis=0)
            y_main=np.append(y_main, y, axis=0)
            alpha = random.choice(np.linspace(0.9, 1.1))
            alpha = 1
            
        indices = np.arange(0,self.batch_size)
        np.random.shuffle(indices)

        return {'images_t0':X_t0_main[indices]/255.0, 'images_tt':X_tt_main[indices]/255.0}, y_main[indices]
        
    def augment(self, X):
#         # Horizontal and vertical flip
#         flipping_funcs = [
#             lambda image: image,
#             lambda image: np.fliplr(image),
#             lambda image: np.flipud(image),
#             lambda image: np.flipud(np.fliplr(image))
#         ]
#         func = random.choice(flipping_funcs)
#         X = func(X)
        
#         # Brightness
#         alpha = random.choice(np.linspace(0.85, 1.4))
# #         alpha = 1
#         X = X * alpha

        return X