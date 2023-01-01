import zarr
from pathlib import Path
import os
import math
import numpy as np
from tensorflow.keras import backend, layers, models, callbacks, metrics
from tensorflow.keras.utils import Sequence
import random
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt
import time
import shutil


CITIES = ['aleppo', 'raqqa']
DATA_DIR = "../data"
OUTPUT_DIR = "../outputs"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cities", help="Cities, comma separated. Eg: aleppo,raqqa,damascus")
args = parser.parse_args()

if args.cities:
    CITIES = [el.strip() for el in args.cities.split(",")]

def read_zarr(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    return zarr.open(path)

def save_zarr(data, path):
    # path = f'{path}/{city}/others/{city}_{suffix}.zarr'

    if not os.path.exists(path):
        zarr.save(path, data)        
    else:
        za = zarr.open(path, mode='a')
        za.append(data)

def delete_zarr_if_exists(path):
    # path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    if os.path.exists(path):
        shutil.rmtree(path)

def make_tuple_pair(n, step_size):
    if step_size > n:
        return [(0,n)]
    iters = math.ceil(n/step_size*1.0)
    l = []
    for i in range(0, iters):
        if i == iters - 1:
            t = (i*step_size, n)
            l.append(t)
        else:
            t = (i*step_size, (i+1)*step_size)
            l.append(t)
    return l


Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

runs = [f for f in os.listdir(OUTPUT_DIR) if ".log" not in f]
runs = [f for f in runs if ".DS_Store" not in f]
run_id = len(runs)+1

print(f"Run ID: {run_id} (use this code for dense_predict.py) \n\n")
time.sleep(5)


RUN_DIR = OUTPUT_DIR + f"/{run_id}"
Path(RUN_DIR).mkdir(parents=True, exist_ok=True)

f = open(f"{OUTPUT_DIR}/runs.log", "a")
f.write(f"Run {run_id}: {CITIES} \n")
f.close()

# im_tr_pre = None
# im_tr_post = None
# la_tr = None
# im_va_pre = None
# im_va_post = None
# la_va = None


for city in CITIES:
    im_tr_pre = read_zarr(city, "im_tr_pre", DATA_DIR)
    im_tr_post = read_zarr(city, "im_tr_post", DATA_DIR)
    la_tr= read_zarr(city, "la_tr", DATA_DIR)

    im_va_pre = read_zarr(city, "im_va_pre", DATA_DIR)
    im_va_post = read_zarr(city, "im_va_post", DATA_DIR)
    la_va = read_zarr(city, "la_va", DATA_DIR)

    im_te_pre = read_zarr(city, "im_te_pre", DATA_DIR)
    im_te_post = read_zarr(city, "im_te_post", DATA_DIR)
    la_te = read_zarr(city, "la_te", DATA_DIR)


    steps = make_tuple_pair(im_tr_pre.shape[0], 5000)
    
    for st in steps:
        _im_tr_pre = im_tr_pre[st[0]:st[1]]
        _im_tr_post = im_tr_post[st[0]:st[1]]
        _la_tr = la_tr[st[0]:st[1]]

        save_zarr(_im_tr_pre, f"{RUN_DIR}/im_tr_pre.zarr")
        save_zarr(_im_tr_post, f"{RUN_DIR}/im_tr_post.zarr")
        save_zarr(_la_tr, f"{RUN_DIR}/la_tr.zarr")

        _im_va_pre = im_va_pre[st[0]:st[1]]
        _im_va_post = im_va_post[st[0]:st[1]]
        _la_va = la_va[st[0]:st[1]]

        save_zarr(_im_va_pre, f"{RUN_DIR}/im_va_pre.zarr")
        save_zarr(_im_va_post, f"{RUN_DIR}/im_va_post.zarr")
        save_zarr(_la_va, f"{RUN_DIR}/la_va.zarr")

        _im_te_pre = im_te_pre[st[0]:st[1]]
        _im_te_post = im_te_post[st[0]:st[1]]
        _la_te = la_te[st[0]:st[1]]

        save_zarr(_im_te_pre, f"{RUN_DIR}/im_te_pre.zarr")
        save_zarr(_im_te_post, f"{RUN_DIR}/im_te_post.zarr")
        save_zarr(_la_te, f"{RUN_DIR}/la_te.zarr")


im_tr_pre = zarr.open(f"{RUN_DIR}/im_tr_pre.zarr")
im_tr_post = zarr.open(f"{RUN_DIR}/im_tr_post.zarr")
la_tr= zarr.open(f"{RUN_DIR}/la_tr.zarr")

im_va_pre = zarr.open(f"{RUN_DIR}/im_va_pre.zarr")
im_va_post = zarr.open(f"{RUN_DIR}/im_va_post.zarr")
la_va = zarr.open(f"{RUN_DIR}/la_va.zarr")

im_te_pre = zarr.open(f"{RUN_DIR}/im_te_pre.zarr")
im_te_post = zarr.open(f"{RUN_DIR}/im_te_post.zarr")
la_te = zarr.open(f"{RUN_DIR}/la_te.zarr")


f = open(f"{RUN_DIR}/metadata.txt", "a")
f.write(f"\n\n######## Run {run_id}: {CITIES} \n\n")
f.write(f"Training Set: {np.unique(la_tr[:], return_counts=True)} \n")
f.write(f"Validation Set: {np.unique(la_va[:], return_counts=True)} \n")
f.write(f"Test Set: {np.unique(la_te[:], return_counts=True)} \n")
f.close()



# Begin SNN Code

BATCH_SIZE = 32
PATCH_SIZE = (128,128)
FILTERS = [8]
DROPOUT = [0.1, 0.15]
EPOCHS = [70, 100]
UNITS = [8]
LR = [0.002, 0.003, 0.004]


def dense_block(inputs, units:int=1, dropout:float=0, name:str=''):
    tensor = layers.Dense(units=units, use_bias=False, kernel_initializer='he_normal', name=f'{name}_dense')(inputs)
    tensor = layers.Activation('relu', name=f'{name}_activation')(tensor)
    tensor = layers.BatchNormalization(name=f'{name}_normalisation')(tensor)
    tensor = layers.Dropout(rate=dropout, name=f'{name}_dropout')(tensor)
    return tensor 

# def convolution_block(inputs, filters:int, dropout:float, name:str):
#     tensor = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_convolution1')(inputs)
#     tensor = layers.Activation('relu', name=f'{name}_activation1')(tensor)
#     tensor = layers.BatchNormalization(name=f'{name}_normalisation1')(tensor)
#     tensor = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_convolution2')(tensor)
#     tensor = layers.Activation('relu', name=f'{name}_activation2')(tensor)
#     tensor = layers.BatchNormalization(name=f'{name}_normalisation2')(tensor)
#     tensor = layers.MaxPool2D(pool_size=(2, 2), name=f'{name}_pooling')(tensor)
#     tensor = layers.SpatialDropout2D(rate=dropout, name=f'{name}_dropout')(tensor)
#     return tensor

def convolution_block(inputs, filters:int, dropout:float, name:str):
    convolution   = layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_convolution')(inputs)
    pooling       = layers.MaxPool2D(pool_size=(2, 2), name=f'{name}_pooling')(convolution)
    normalisation = layers.BatchNormalization(name=f'{name}_normalisation')(pooling)
    outputs       = layers.SpatialDropout2D(rate=dropout, name=f'{name}_dropout')(normalisation)
    return outputs


def distance_layer(inputs):
    input0, input1 = inputs
    distances = tf.math.reduce_sum(tf.math.square(input0 - input1), axis=1, keepdims=True)
    distances = tf.math.sqrt(tf.math.maximum(distances, tf.keras.backend.epsilon()))
    return distances



def encoder_block_separated(inputs, filters:int=1, dropout=0, name:str=''):
    tensor  = convolution_block(inputs, filters=filters*1, dropout=dropout, name=f'{name}_block1')
    tensor  = convolution_block(tensor, filters=filters*2, dropout=dropout, name=f'{name}_block2')
    tensor  = convolution_block(tensor, filters=filters*3, dropout=dropout, name=f'{name}_block3')
    tensor  = convolution_block(tensor, filters=filters*4, dropout=dropout, name=f'{name}_block4')
    tensor  = convolution_block(tensor, filters=filters*5, dropout=dropout, name=f'{name}_block5')
    outputs = layers.Flatten(name=f'{name}_flatten')(tensor)
    return outputs


def siamese_convolutional_network(shape:tuple, args_encode:dict, args_dense:dict):
    # Input layers
    images1 = layers.Input(shape=shape, name='images_t0')
    images2 = layers.Input(shape=shape, name='images_tt')
    # Hidden convolutional layers (shared parameters)
    encoder_block = encoder_block_shared(shape=shape, **args_encode)
    encode1 = encoder_block(images1)
    encode2 = encoder_block(images2)
    # Hidden dense layers
    distance = distance_layer([encode1, encode2])
    # concat  = layers.Concatenate(name='concatenate')(inputs=[encode1, encode2])
    dense   = dense_block(distance, **args_dense, name='dense_block1')
    dense   = dense_block(dense,    **args_dense, name='dense_block2')
    dense   = dense_block(dense,    **args_dense, name='dense_block3')
    # Output layer
    outputs = layers.Dense(units=1, activation='sigmoid', name='outputs')(dense)
    # Model
    model   = models.Model(inputs=[images1, images2], outputs=outputs, name='siamese_convolutional_network')
    return model


def double_convolutional_network(shape:tuple, args_encode:dict, args_dense:dict):
    # Input layers
    images1 = layers.Input(shape=shape, name='images_t0')
    images2 = layers.Input(shape=shape, name='images_tt')
    # Hidden convolutional layers (shared parameters)
    encode1 = encoder_block_separated(images1, **args_encode, name='encoder1')
    encode2 = encoder_block_separated(images2, **args_encode, name='encoder2')
    # Hidden dense layers
    concat  = layers.Concatenate(name='concatenate')(inputs=[encode1, encode2])
    dense   = dense_block(concat, **args_dense, name='dense_block1')
    dense   = dense_block(dense,  **args_dense, name='dense_block2')
    dense   = dense_block(dense,  **args_dense, name='dense_block3')
    # Output layer
    outputs = layers.Dense(units=1, activation='sigmoid', name='outputs')(dense)
    # Model
    model   = models.Model(inputs=[images1, images2], outputs=outputs, name='siamese_convolutional_network')
    return model

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


gen_tr = SiameseGenerator((im_tr_pre, im_tr_post), la_tr)
gen_va = SiameseGenerator((im_va_pre, im_va_post), la_va)

print("+++++++++", gen_tr.__len__())
MODEL_STORAGE_LOCATION = f"{RUN_DIR}/model"
training_callbacks = [
    callbacks.EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint(f'{MODEL_STORAGE_LOCATION}', monitor='val_auc', verbose=0, save_best_only=True, save_weights_only=False, mode='max')
]


filters = random.choice(FILTERS)
dropout = random.choice(np.linspace(DROPOUT[0], DROPOUT[1]))
epochs = random.choice(np.arange(EPOCHS[0],EPOCHS[1]))
units = random.choice(UNITS)
lr = random.choice(LR)

args  = dict(filters=filters, dropout=dropout, units=units) # ! Check parameters before run


args_encode = dict(filters=filters, dropout=dropout)
args_dense  = dict(units=units, dropout=dropout)
parameters = f'filters={filters}, \ndropout={np.round(dropout, 4)}, \nepochs={epochs}, \nunits={units}, \nlearning_rate={lr}'
print(parameters)
f = open(f"{RUN_DIR}/metadata.txt", "a")
f.write(f"\n######## Run parameters \n\n{parameters}")
f.close()

model = siamese_convolutional_network(
    shape=(*PATCH_SIZE, 3),  
    args_encode = args_encode,
    args_dense = args_dense,
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',metrics.AUC(num_thresholds=200, curve='ROC', name='auc')])
model.summary()


try:
  history = model.fit(
    gen_tr,
    validation_data=gen_va,
    epochs=epochs,
    verbose=1,
    callbacks=training_callbacks)
except:
  print("## Model training stopped, generating numbers on best model so far..")
  print("## Please wait, the program will terminate automatically..")
# Train model on dataset


def plot_training(H):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["accuracy"], label="train_accuracy")
	plt.plot(H.history["val_accuracy"], label="val_accuracy")
	plt.plot(H.history["auc"], label="train_auc")
	plt.plot(H.history["val_auc"], label="val_auc")
	plt.title(f"Training Accuracy and AUC")
	plt.suptitle(f"Cities = {CITIES}; RUN ID = {run_id}") 
	plt.xlabel("Epoch #")
	plt.ylabel("AUC")
	plt.text(0.65, 0.18, f"\nmax(val_auc)={np.round(np.max(H.history['val_auc']), 4)}", fontsize=8, transform=plt.gcf().transFigure)
	plt.legend(loc="lower left")
	plt.savefig(f"{RUN_DIR}/training.png")

plot_training(history)

# model_path = f'{MODEL_DIR}/{CITY}/snn/run_{i}'
best_model = load_model(MODEL_STORAGE_LOCATION, custom_objects={'auc':metrics.AUC(num_thresholds=200, curve='ROC', name='auc')})
gen_te= SiameseGenerator((im_te_pre, im_te_post), la_te, train=False)
yhat_proba, y = np.squeeze(best_model.predict(gen_te)), np.squeeze(la_te[0:(len(la_te)//BATCH_SIZE)*BATCH_SIZE])
roc_auc_test = roc_auc_score(y, yhat_proba)
#calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y, yhat_proba)


#create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')

#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
f = open(f"{RUN_DIR}/metadata.txt", "a")
f.write("\n\n######## Test set performance\n\n")
f.write(f'Test Set AUC Score for the ROC Curve: {roc_auc_test} \nAverage precision:  {np.mean(precision)}')
f.close()
#display plot
plt.savefig(f"{RUN_DIR}/pr_curve.png")

delete_zarr_if_exists(f"{RUN_DIR}/im_tr_pre.zarr")
delete_zarr_if_exists(f"{RUN_DIR}/im_tr_post.zarr")
delete_zarr_if_exists(f"{RUN_DIR}/la_tr.zarr")

delete_zarr_if_exists(f"{RUN_DIR}/im_va_pre.zarr")
delete_zarr_if_exists(f"{RUN_DIR}/im_va_post.zarr")
delete_zarr_if_exists(f"{RUN_DIR}/la_va.zarr")

delete_zarr_if_exists(f"{RUN_DIR}/im_te_pre.zarr")
delete_zarr_if_exists(f"{RUN_DIR}/im_te_post.zarr")
delete_zarr_if_exists(f"{RUN_DIR}/la_te.zarr")





