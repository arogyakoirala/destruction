import numpy as np
from tensorflow.keras import callbacks, metrics
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import time
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_curve, roc_auc_score, precision_score, recall_score
import zarr
import random
from tensorflow.keras import backend, layers, models
from tensorflow import math
from datetime import datetime


CITY = 'aleppo_cropped'
SUFFIX = 'im'
N_BANDS = 3
DATA_DIR = "../../../data"
MODEL_DIR = "../../../models"
BATCH_SIZE = 64
PATCH_SIZE = (128,128)
FILTERS = [32]
DROPOUT = [0.3, 0.45]
EPOCHS = [70, 100]
UNITS = [64]
LR = [0.003, 0.003, 0.004]


def read_zarr(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    return zarr.open(path)

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


class SiameseGenerator(Sequence):
    def __init__(self, images, labels, batch_size=32):
        self.images_pre = images[0]
        self.images_post = images[1]
        self.labels = labels
        self.batch_size = batch_size
        
        # self.tuple_pairs = make_tuple_pair(self.images_t0.shape[0], int(self.batch_size/4))
        # np.random.shuffle(self.tuple_pairs)
    def __len__(self):
        return len(self.images_pre)//self.batch_size    
    
    def __getitem__(self, index):

        X_pre = self.images_pre[index*self.batch_size:(index+1)*self.batch_size]
        X_post = self.images_post[index*self.batch_size:(index+1)*self.batch_size]
        y = self.labels[index*self.batch_size:(index+1)*self.batch_size]

        return {'images_t0': X_pre, 'images_tt': X_post}, y

im_tr_pre= read_zarr(CITY, f'{SUFFIX}_tr_pre')
im_tr_post = read_zarr(CITY, f'{SUFFIX}_tr_post')
la_tr = read_zarr(CITY, 'la_tr')

im_va_pre= read_zarr(CITY, f'{SUFFIX}_va_pre')
im_va_post = read_zarr(CITY, f'{SUFFIX}_va_post')
la_va = read_zarr(CITY, 'la_va')

im_te_pre= read_zarr(CITY, f'{SUFFIX}_te_pre')
im_te_post = read_zarr(CITY, f'{SUFFIX}_te_post')
la_te = read_zarr(CITY, 'la_te')

gen_tr = SiameseGenerator((im_tr_pre, im_tr_post), la_tr)
gen_va = SiameseGenerator((im_va_pre, im_va_post), la_va)

MODEL_STORAGE_LOCATION = f"{MODEL_DIR}/snn_{SUFFIX}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"

print(f"Label distributions: {np.unique(la_tr[:], return_counts=True )}")

training_callbacks = [
    callbacks.EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True),
    callbacks.ModelCheckpoint(f'{MODEL_STORAGE_LOCATION}', monitor='val_auc', verbose=0, save_best_only=True, save_weights_only=False, mode='max')
]


filters = random.choice(FILTERS)
dropout = random.choice(np.linspace(DROPOUT[0], DROPOUT[1]))
epochs = random.choice(np.arange(EPOCHS[0],EPOCHS[1]))
units = random.choice(UNITS)
lr = random.choice(LR)

args  = dict(filters=filters, dropout=dropout, units=units) # ! Check parameters before run
parameters = f'filters={filters}, \ndropout={np.round(dropout, 4)}, \nepochs={epochs}, \nunits={units}, \nlearning_rate={lr}'
print(parameters)

model = models.siamese_convolutional_network_dist(
        shape=(*PATCH_SIZE, 3),  
        args = args,
    )
   
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',metrics.AUC(num_thresholds=200, curve='ROC', name='auc')])
model.summary()


# Train model on dataset
history = model.fit(
    gen_tr,
    validation_data=gen_va,
    epochs=epochs,
    verbose=1,
    callbacks=training_callbacks
)