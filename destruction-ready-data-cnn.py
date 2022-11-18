import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--city", help="city")
parser.add_argument('--data_dir', help="Path to data folder")
parser.add_argument("--pre_image_index", help="index of images to use as pre image")
parser.add_argument('--window', default=False, action="store_true",
                    help="Use a window?")
parser.add_argument('--window_size', help="Size of window (comma separated, eg: 20,20")
parser.add_argument("--dataset", help="One of: test, train, validate")
parser.add_argument('--balance', default=False, action="store_true", help="Balance (upsample) positives in data?")
parser.add_argument("--patch_size", help="Patch Size")

args = parser.parse_args()

import os
import sys
import time
import gc
import numpy as np

from scripts import destruction_utilities as utils

DEBUG = False
CITY = 'aleppo'
DATA_DIR = "../data"
PRE_IMAGE_INDEX=[0]
WINDOW = False
WINDOW_SIZE = (40,20)
DATASET = 'all'
BALANCE=False
TILE_SIZE = (128,128)

if args.city:
    CITY=args.city

if args.data_dir:
    DATA_DIR=args.data_dir
    
if args.pre_image_index:
    PRE_IMAGE_INDEX = [int(el.strip()) for el in args.pre_image_index.split(",")]

if args.window:
    WINDOW = args.window

if args.window_size:
    WINDOW_SIZE = [int(el.strip()) for el in args.window_size.split(",")]

if args.dataset:
    DATASET = args.dataset

if args.balance:
    BALANCE = args.balance
    
if args.patch_size:
    TILE_SIZE = [int(el.strip()) for el in args.patch_size.split(",")]

print(f"Parameters: city={CITY}, data_dir={DATA_DIR}, pre_image_index={PRE_IMAGE_INDEX}, window={WINDOW}, window_size={WINDOW_SIZE}, dataset={DATASET}, balance={BALANCE}, tile_size={TILE_SIZE}")

if WINDOW:
    window = utils.center_window(f'{DATA_DIR}/{CITY}/others/{CITY}_samples.tif', (WINDOW_SIZE[0]*1, WINDOW_SIZE[1]*1))
    samples = utils.read_raster(f'{DATA_DIR}/{CITY}/others/{CITY}_samples.tif', window=window)
else:
    samples = utils.read_raster(f'{DATA_DIR}/{CITY}/others/{CITY}_samples.tif')
images  = utils.search_data(utils.pattern(city=CITY, type='image'), directory=DATA_DIR)
labels  = utils.search_data(utils.pattern(city=CITY, type='label'), directory=DATA_DIR)

if DATASET == 'train' or DATASET=='all':
    utils.delete_zarr_if_exists(CITY, 'labels_conv_train', path=DATA_DIR)
    utils.delete_zarr_if_exists(CITY, 'images_conv_train', path=DATA_DIR)
if DATASET == 'validate' or DATASET=='all':
    utils.delete_zarr_if_exists(CITY, 'labels_conv_valid', path=DATA_DIR)
    utils.delete_zarr_if_exists(CITY, 'images_conv_valid', path=DATA_DIR)
if DATASET == 'test' or DATASET=='all':
    utils.delete_zarr_if_exists(CITY, 'labels_conv_test', path=DATA_DIR)
    utils.delete_zarr_if_exists(CITY, 'images_conv_test', path=DATA_DIR)
    
image_dates = sorted([el.split("image_")[1].split('.tif')[0] for el in images])
label_dates = sorted([el.split("label_")[1].split('.tif')[0] for el in labels])

# Map labels and images
for label in label_dates:
    if label.replace("-", "_") not in image_dates:
        latest_available_image = sorted([im for im in image_dates if time.strptime(im, "%Y_%m_%d")  < time.strptime(label, "%Y-%m-%d")])
        latest_available_image = latest_available_image[-1]
        if DEBUG:
            print(label, latest_available_image)
        images.append(images[0].split("image_")[0]+"image_"+latest_available_image+".tif")
images = sorted(images)

for i in range(len(images)):
    if WINDOW:
        window = utils.center_window(labels[i], (WINDOW_SIZE[0]*1, WINDOW_SIZE[1]*1))
        label = np.array(utils.read_raster(labels[i], window=window))
    else:
        label = np.array(utils.read_raster(labels[i]))

    h,w,b = label.shape
    label = label.reshape(1, h, w, b)
    label = utils.tile_sequences(label, tile_size=(1,1))
    exclude = np.where(label.flatten() == -1)
    label = np.delete(label, exclude, 0)
    label[label!=3.0] = 0.0
    label[label==3.0] = 1.0
    _, train, validate, test = utils.sample_split(label, np.delete(samples.flatten(), exclude))
    
        
    if DATASET == 'train' or DATASET=='all':
            train_shuffle = np.arange(len(train))
            np.random.shuffle(train_shuffle)
            utils.save_zarr(train[train_shuffle].reshape(np.take(train.shape, [0,2,3,4])), CITY, 'labels_conv_train', path=DATA_DIR)

    if DATASET == 'validate' or DATASET=='all':
        validate_shuffle = np.arange(len(validate))
        np.random.shuffle(validate_shuffle)
        utils.save_zarr(validate[validate_shuffle].reshape(np.take(validate.shape, [0,2,3,4])), CITY, 'labels_conv_valid', path=DATA_DIR)

    if DATASET == 'test' or DATASET=='all':
        test_shuffle = np.arange(len(test))
        np.random.shuffle(test_shuffle)
        utils.save_zarr(test[test_shuffle].reshape(np.take(test.shape, [0,2,3,4])), CITY, 'labels_conv_test', path=DATA_DIR)
        
    del _, train, validate, test, label
    
    if WINDOW:
        window = utils.center_window(images[i], (WINDOW_SIZE[0]*TILE_SIZE[0], WINDOW_SIZE[1]*TILE_SIZE[1]))
        image = np.array(utils.read_raster(images[i], window=window))
    else:
        image = np.array(utils.read_raster(images[i]))
        
    h,w,b = image.shape
    image = image.reshape(1,h,w,b)
    image = utils.tile_sequences(image,  tile_size=TILE_SIZE)
    image = np.delete(image, exclude, 0)
    dum_ = image # comment in prod
    
    _, train, validate, test = utils.sample_split(image, np.delete(samples.flatten(), exclude))
    if DEBUG:
        print("New Image Shape:", image.shape)
        
    if DATASET == 'train' or DATASET=='all':
        utils.save_zarr(train[train_shuffle].reshape(np.take(train.shape, [0,2,3,4])), CITY, 'images_conv_train', path=DATA_DIR)
    if DATASET == 'validate' or DATASET=='all':
        utils.save_zarr(validate[validate_shuffle].reshape(np.take(validate.shape, [0,2,3,4])), CITY,'images_conv_valid', path=DATA_DIR)
    if DATASET == 'test' or DATASET=='all':
        utils.save_zarr(test[test_shuffle].reshape(np.take(test.shape, [0,2,3,4])), CITY,'images_conv_test', path=DATA_DIR) 
    del _, train, validate, test, image, exclude
    print(f'------ {label_dates[i]}')

    gc.collect(generation=2)
del samples, images, labels

if DATASET == 'train' or DATASET=='all':
    #%% 
    # Generate a balanced (upsampled) dataset and shuffle it..
    utils.delete_zarr_if_exists(CITY, 'labels_conv_train_balanced', path=DATA_DIR)
    utils.delete_zarr_if_exists(CITY, 'images_conv_train_balanced', path=DATA_DIR)
    utils.delete_zarr_if_exists(CITY, 'labels_conv_train_balanced_shuffled', path=DATA_DIR)
    utils.delete_zarr_if_exists(CITY, 'images_conv_train_balanced_shuffled', path=DATA_DIR)
    if BALANCE:
        print('\n--- Generate a balanced (upsampled) dataset..')
        utils.balance(CITY, path=DATA_DIR)
    print('--- Shuffle dataset..')
    utils.shuffle(CITY, TILE_SIZE, (100,750), path=DATA_DIR)

print(f"--- Data prep complete for {CITY}")

