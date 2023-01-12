import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--train", help="What city do we use for training and validation data?")
parser.add_argument("--test", help="What city do we use for test data?")
parser.add_argument("--model_dir", help="Where will output models be stored?")
parser.add_argument("--data_dir", help="Root directory for data folder?")
parser.add_argument("--batch_size", type=int, help="Batch size")
parser.add_argument("--patch_size", type=int, help="Patch size, e.g 128")
parser.add_argument("--filters", help="Number of filters to try, e.g. 32,64")
parser.add_argument("--range_dropout",  help="Range for dropout, eg, 0.25,0.35")
parser.add_argument("--range_epochs",  help="Number of filters")
parser.add_argument("--units",  help="Number of units to try, e.g. 32,64")
parser.add_argument("--lr",  help="Different learning rates to try, e.g. 32,64")
parser.add_argument("--variant",  help="What variant of the model should we use?")
parser.add_argument("--patience", type=int, help="Patience, i.e. how many epochs to wait before I kill the model run?")

args = parser.parse_args()


import os
import sys
import random
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import callbacks, metrics
import numpy as np

from scripts import destruction_utilities as utils
from scripts import destruction_models_cnn as cnn
from scripts import destruction_models_snn as snn

print(utils)

TRAIN      = 'aleppo'
TEST       = 'aleppo'
MODEL_DIR  = '../models'
DATA_DIR   = '../data'
BATCH_SIZE = 32
PATCH_SIZE = (128,128)
FILTERS    = [8]
DROPOUT    = [0.1, 0.15]
EPOCHS     = [70, 100]
UNITS      = [32]
LR         = [0.003, 0.003, 0.004]
VARIANT    = "chained_snn" # snn, cnn
PATIENCE    = 8 # snn, cnn

if args.train:
    TRAIN = args.train

if args.test:
    TEST = args.test

if args.model_dir:
    MODEL_DIR = args.model_dir

if args.data_dir:
    DATA_DIR = args.data_dir

if args.batch_size:
    BATCH_SIZE = args.batch_size

if args.patch_size:
    PATCH_SIZE = (args.patch_size, args.patch_size)

if args.filters:
    FILTERS = [int(el.strip()) for el in args.filters.split(",")]

if args.range_dropout:
    DROPOUT = [int(el.strip()) for el in args.range_dropout.split(",")]
    assert len(DROPOUT) == 2

if args.range_epochs: 
    EPOCHS = [int(el.strip()) for el in args.range_epochs.split(",")][:2]
    assert len(EPOCHS) == 2

if args.units:
    UNITS = [int(el.strip()) for el in args.units.split(",")]

if args.lr:
    LR = [int(el.strip()) for el in args.lr.split(",")]

if args.variant:
    VARIANT = args.variant

if args.patience:
    PATIENCE = args.patience
    
print(f"\nParameters: \n\nTRAIN={TRAIN}, TEST={TEST}, MODEL_DIR={MODEL_DIR}, DATA_DIR={DATA_DIR}, BATCH_SIZE={BATCH_SIZE}, PATCH_SIZE={PATCH_SIZE}, FILTERS={FILTERS}, DROPOUT={DROPOUT}, EPOCHS={EPOCHS}, UNITS={UNITS}, LR={LR}, VARIANT={VARIANT}\n\n")


if VARIANT in ['chained_snn', 'snn']:
    train_images_t0 = utils.read_zarr(TRAIN, 'images_siamese_train_t0', path=DATA_DIR)
    train_images_tt = utils.read_zarr(TRAIN, 'images_siamese_train_tt', path=DATA_DIR)
    train_labels    = utils.read_zarr(TRAIN, 'labels_siamese_train', path=DATA_DIR)

    valid_images_t0 = utils.read_zarr(TRAIN, 'images_siamese_valid_t0', path=DATA_DIR)
    valid_images_tt = utils.read_zarr(TRAIN, 'images_siamese_valid_tt', path=DATA_DIR)
    valid_labels    = utils.read_zarr(TRAIN, 'labels_siamese_valid', path=DATA_DIR)


    test_images_t0  = utils.read_zarr(TEST, 'images_siamese_test_t0', path=DATA_DIR)
    test_images_tt  = utils.read_zarr(TEST, 'images_siamese_test_tt', path=DATA_DIR)
    test_labels     = utils.read_zarr(TEST, 'labels_siamese_test', path=DATA_DIR)

    train_gen       = utils.SiameseGenerator((train_images_t0, train_images_tt), train_labels, batch_size=BATCH_SIZE)
    valid_gen       = utils.SiameseGenerator((valid_images_t0, valid_images_tt), valid_labels, batch_size=BATCH_SIZE)

if VARIANT == 'cnn':
    train_images = utils.read_zarr(TRAIN, 'images_conv_train', path=DATA_DIR)
    train_labels = utils.read_zarr(TRAIN, 'labels_conv_train', path=DATA_DIR)
    valid_images = utils.read_zarr(TRAIN, 'images_conv_valid', path=DATA_DIR)
    valid_labels = utils.read_zarr(TRAIN, 'labels_conv_valid', path=DATA_DIR)
    test_images = utils.read_zarr(TEST, 'images_conv_test', path=DATA_DIR)
    test_labels = utils.read_zarr(TEST, 'labels_conv_test', path=DATA_DIR)

    train_gen = utils.CNNGenerator(train_images, train_labels, batch_size=BATCH_SIZE)
    valid_gen = utils.CNNGenerator(valid_images, valid_labels, batch_size=BATCH_SIZE)

print(f"Label distributions: {np.unique(train_labels[:], return_counts=True )}")

MODEL_STORAGE_LOCATION = f"{MODEL_DIR}/{VARIANT}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"

training_callbacks = [
    callbacks.EarlyStopping(monitor='val_auc', patience=PATIENCE, restore_best_weights=True),
    callbacks.ModelCheckpoint(f'{MODEL_STORAGE_LOCATION}', monitor='val_auc', verbose=0, save_best_only=True, save_weights_only=False, mode='max')
]



print(f"Model will be stored at: {MODEL_STORAGE_LOCATION}")

filters = random.choice(FILTERS)
dropout = random.choice(np.linspace(*DROPOUT))
epochs  = random.choice(np.arange(*EPOCHS))
units   = random.choice(UNITS)
lr      = random.choice(LR)



if VARIANT == 'snn':
    args_encode = dict(filters=filters, dropout=dropout)
    args_dense  = dict(units=units, dropout=dropout)
    parameters  = f'Model parameters: filters={filters}, \ndropout={np.round(dropout, 4)}, \nepochs={epochs}, \nunits={units}, \nlearning_rate={lr}'

    model = snn.siamese_convolutional_network(
        shape=(*PATCH_SIZE, 3),  
        args_encode = args_encode,
        args_dense = args_dense
    )

elif VARIANT == 'chained_snn':

    args_encode = dict(filters=filters, dropout=dropout)
    args_dense  = dict(units=units, dropout=dropout)
    parameters  = f'Model parameters: filters={filters}, \ndropout={np.round(dropout, 4)}, \nepochs={epochs}, \nunits={units}, \nlearning_rate={lr}'

    model = snn.double_convolutional_network(
        shape=(*PATCH_SIZE, 3),  
        args_encode = args_encode,
        args_dense = args_dense,
    )

elif VARIANT == 'cnn':
    args  = dict(shape=(*PATCH_SIZE, 3), filters=filters, units=units, dropout=dropout)
    model = cnn.convolutional_network(**args)
        
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer, loss='binary_focal_crossentropy', metrics=['accuracy', metrics.Precision(thresholds=0.5), metrics.AUC(num_thresholds=200, curve='ROC', name='auc')])


model.summary()
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=epochs,
    verbose=1,
    callbacks=training_callbacks
)

