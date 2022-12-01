import numpy as np
from tensorflow.keras import callbacks, metrics
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import time
import pickle
from keras.models import load_model
from sklearn.metrics import precision_recall_curve, roc_auc_score
import zarr
import random
from tensorflow.keras import backend, layers, models
from tensorflow import math
from datetime import datetime
import matplotlib.pyplot as plt
import os
import re
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, AUC
from tensorflow.keras import backend as K
import rasterio
import shutil
from sklearn.metrics import classification_report, confusion_matrix


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--suffix", help="Suffix")
parser.add_argument("--bands", help="Numper of bands", type=int)
args = parser.parse_args()


CITY = 'aleppo_cropped'
SUFFIX = 'lap'
BANDS = 1
# N_BANDS = 3
DATA_DIR = "../../../data"
MODEL_DIR = "../../../models"
BATCH_SIZE = 32
PATCH_SIZE = (128,128)
FILTERS = [8]
DROPOUT = [0.1, 0.15]
EPOCHS = [70, 100]
UNITS = [8]
LR = [0.003, 0.003, 0.004]

if args.suffix:
    SUFFIX = args.suffix

if args.bands:
    BANDS = args.bands

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

def tiled_profile(source:str, tile_size:tuple=(*PATCH_SIZE, 1)) -> dict:
    '''Computes raster profile for tiles'''
    raster  = rasterio.open(source)
    profile = raster.profile
    assert profile['width']  % tile_size[0] == 0, 'Invalid dimensions'
    assert profile['height'] % tile_size[1] == 0, 'Invalid dimensions'
    affine  = profile['transform']
    affine  = rasterio.Affine(affine[0] * tile_size[0], affine[1], affine[2], affine[3], affine[4] * tile_size[1], affine[5])
    profile.update(width=profile['width'] // tile_size[0], height=profile['height'] // tile_size[0], count=tile_size[2], transform=affine)
    return profile

def search_data(pattern:str='.*', directory:str='../data') -> list:
    '''Sorted list of files in a directory matching a regular expression'''
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    if len(files) == 1: files = files[0]
    return files

def pattern(city:str='.*', type:str='.*', date:str='.*', ext:str='tif') -> str:
    '''Regular expressions for search_data'''
    return f'^.*{city}/.*/{type}_{date}\.{ext}$'

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def read_raster(source:str, band:int=None, window=None, dtype:str='int', profile:bool=False) -> np.ndarray:
    '''Reads a raster as a numpy array'''
    raster = rasterio.open(source)
    if band is not None:
        image = raster.read(band, window=window)
        image = np.expand_dims(image, 0)
    else: 
        image = raster.read(window=window)
    image = image.transpose([1, 2, 0]).astype(dtype)
    if profile:
        return image, raster.profile
    else:
        return image


def tile_sequences(images:np.ndarray, tile_size:tuple=(128, 128)) -> np.ndarray:
    '''Converts images to sequences of tiles'''
    n_images, image_height, image_width, n_bands = images.shape
    tile_width, tile_height = tile_size
    assert image_width  % tile_width  == 0
    assert image_height % tile_height == 0
    n_tiles_width  = (image_width  // tile_width)
    n_tiles_height = (image_height // tile_height)
    sequence = images.reshape(n_images, n_tiles_width, tile_width, n_tiles_height, tile_height, n_bands)
    sequence = np.moveaxis(sequence.swapaxes(2, 3), 0, 2)
    sequence = sequence.reshape(-1, n_images, tile_width, tile_height, n_bands)
    return sequence


im_tr_pre= read_zarr(CITY, f'{SUFFIX}_tr_pre', path=DATA_DIR)[:]
im_tr_post = read_zarr(CITY, f'{SUFFIX}_tr_post', path=DATA_DIR)[:]
la_tr = read_zarr(CITY, 'la_tr', path=DATA_DIR)[:]

print(f"Training Set: {la_tr.shape[0]} observations")

im_va_pre= read_zarr(CITY, f'{SUFFIX}_va_pre', path=DATA_DIR)[:]
im_va_post = read_zarr(CITY, f'{SUFFIX}_va_post', path=DATA_DIR)[:]
la_va = read_zarr(CITY, 'la_va', path=DATA_DIR)[:]

print(f"Validation Set: {la_va.shape[0]} observations")


im_te_pre= read_zarr(CITY, f'{SUFFIX}_te_pre', path=DATA_DIR)[:]
im_te_post = read_zarr(CITY, f'{SUFFIX}_te_post', path=DATA_DIR)[:]
la_te = read_zarr(CITY, 'la_te', path=DATA_DIR)[:]

print(f"Test Set: {la_te.shape[0]} observations")


auc = AUC(
    num_thresholds=200,
    curve='ROC',
    name="auc"
)

# gen_tr = SiameseGenerator((im_tr_pre, im_tr_post), la_tr)
# gen_va = SiameseGenerator((im_va_pre, im_va_post), la_va)

print(im_tr_pre)

MODEL_CODE = f"snn_{SUFFIX}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
MODEL_STORAGE_LOCATION = f"{MODEL_DIR}/{MODEL_CODE}"

# print(f"Label distributions: {np.unique(la_tr[:], return_counts=True )}")

training_callbacks = [
    callbacks.EarlyStopping(monitor='val_auc', patience=4, restore_best_weights=True),
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

args_encode = dict(filters=filters, dropout=dropout)
args_dense  = dict(units=units, dropout=dropout)
parameters  = f'Model parameters: filters={filters}, \ndropout={np.round(dropout, 4)}, \nepochs={epochs}, \nunits={units}, \nlearning_rate={lr}'

model = double_convolutional_network(
    shape=(*PATCH_SIZE, BANDS),  
    args_encode = args_encode,
    args_dense = args_dense,
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',metrics.AUC(num_thresholds=200, curve='ROC', name='auc')])
model.summary()


# Train model on dataset
history = model.fit(
    {'images_t0': im_tr_pre, 'images_tt': im_tr_post}, la_tr,
    validation_data=({'images_t0': im_va_pre, 'images_tt': im_va_post}, la_va),
    epochs=epochs,
    verbose=1,
    callbacks=training_callbacks
)


class SiameseGenerator(Sequence):
    def __init__(self, images, labels, batch_size=32, train=True):
        self.images_pre = images[0]
        self.images_post = images[1]
        self.labels = labels
        self.batch_size = batch_size
        self.train = train


        
        # self.tuple_pairs = make_tuple_pair(self.images_t0.shape[0], int(self.batch_size/4))
        # np.random.shuffle(self.tuple_pairs)
    def __len__(self):
        return len(self.images_pre)//self.batch_size    
    
    def __getitem__(self, index):
        X_pre = self.images_pre[index*self.batch_size:(index+1)*self.batch_size]
        X_post = self.images_post[index*self.batch_size:(index+1)*self.batch_size]
        y = self.labels[index*self.batch_size:(index+1)*self.batch_size]

        if self.train:
            return {'images_t0': X_pre, 'images_tt': X_post}, y
        else:
            return {'images_t0': X_pre, 'images_tt': X_post}



# model_path = f'{MODEL_DIR}/{CITY}/snn/run_{i}'
best_model = load_model(MODEL_STORAGE_LOCATION, custom_objects={'auc':metrics.AUC(num_thresholds=200, curve='ROC', name='auc')})
gen_te= SiameseGenerator((im_te_pre, im_te_post), la_te, train=False)
yhat_proba, y = np.squeeze(best_model.predict(gen_te)), np.squeeze(la_te[0:(len(la_te)//BATCH_SIZE)*BATCH_SIZE])
yhat = np.where(yhat_proba < 0.5, 0, 1)

roc_auc_test = roc_auc_score(y, yhat_proba)
#calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y, yhat_proba)
# F1 = 2 * (precision * recall) / (precision + recall)
# p_score = precision_score(y, yhat_proba)
# r_score = recall_score(y, yhat_proba)
# plot_roc_curve(fpr, tpr)

print(classification_report(y, yhat))

#create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')

#add axis labels to plot
ax.set_title('Precision - Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

print(f'Run {MODEL_STORAGE_LOCATION}: \nTest Set AUC Score for the ROC Curve: {roc_auc_test} \nAverage precision:  {np.mean(precision)}' )
#display plot
plt.savefig(f"{MODEL_DIR}/pr_curve_{SUFFIX}.png")


# images = search_data(pattern(city=CITY, type='image'))
# best_model = load_model(MODEL_STORAGE_LOCATION, custom_objects={'f1_m':f1_m, 'precision_m': precision_m, 'recall_m': recall_m, 'auc': auc})

# images = [images]

# for image in images:
#     profile = tiled_profile(image, tile_size=(*PATCH_SIZE, 1))
#     name = image.split("image_")[1]
#     image = np.array(read_raster(image))
#     h,w,b = image.shape
#     image = image.reshape(1,h,w,b)
#     image = tile_sequences(image, tile_size=PATCH_SIZE)   
#     image = image.reshape(image.shape[0], image.shape[2], image.shape[3], image.shape[4])
#     if os.path.exists('temp.zarr'):
#         shutil.rmtree("temp.zarr")
#     zarr.save("temp.zarr", np.empty((0,image.shape[1], image.shape[2], image.shape[3])))
#     temp = zarr.open("temp.zarr", mode = 'a')
#     temp.append(image)
    
#     # test_images = CNNTestGenerator(temp)
#     labels = best_model.predict(temp)