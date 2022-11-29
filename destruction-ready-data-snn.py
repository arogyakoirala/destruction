import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--city", help="city")
parser.add_argument('--data_dir', help="Path to data folder")
parser.add_argument("--pre_image_index", help="index of images to use as pre image")
parser.add_argument('--window', default=False, action="store_true",
                    help="Use a window?")
parser.add_argument('--window_size', help="Size of window (comma separated, eg: 20,20")
parser.add_argument("--dataset", help="One of: test, train, validate, all")
parser.add_argument('--balance', default=False, action="store_true", help="Balance (upsample) positives in data?")
parser.add_argument("--patch_size", help="Patch Size")
parser.add_argument("--downsample_factor", help="Downsampling factor (default = 1)")
parser.add_argument('--hog', default=False, action="store_true", help="Use HOGs instaed of original image?")

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
PRE_IMAGE_INDEX=[0,1]
WINDOW = False
WINDOW_SIZE = (20,20)
DATASET = 'all'
BALANCE=False
TILE_SIZE = (128,128)
HOG = False

XOFFSET = 85
YOFFSET = -45

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

if args.hog:
    HOG = args.hog

print(f"Parameters: city={CITY}, data_dir={DATA_DIR}, pre_image_index={PRE_IMAGE_INDEX}, window={WINDOW}, window_size={WINDOW_SIZE}, dataset={DATASET}, balance={BALANCE}, tile_size={TILE_SIZE}, hog={HOG}")

if WINDOW:
    window = utils.center_window(f'{DATA_DIR}/{CITY}/others/{CITY}_samples.tif', (WINDOW_SIZE[0]*1, WINDOW_SIZE[1]*1), (1,1), xoffset=XOFFSET, yoffset=YOFFSET)
    samples = utils.read_raster(f'{DATA_DIR}/{CITY}/others/{CITY}_samples.tif', window=window)
else:
    samples = utils.read_raster(f'{DATA_DIR}/{CITY}/others/{CITY}_samples.tif')

images  = utils.search_data(utils.pattern(city=CITY, type='image'), directory=DATA_DIR)
labels  = utils.search_data(utils.pattern(city=CITY, type='label'), directory=DATA_DIR)

if DATASET=='train' or DATASET=='all':
    utils.delete_zarr_if_exists(CITY, 'labels_siamese_train', path=DATA_DIR)
    utils.delete_zarr_if_exists(CITY, 'images_siamese_train_tt', path=DATA_DIR)
    utils.delete_zarr_if_exists(CITY, 'images_siamese_train_t0', path=DATA_DIR)

if DATASET=='validate' or DATASET=='all':
    utils.delete_zarr_if_exists(CITY, 'labels_siamese_valid', path=DATA_DIR)
    utils.delete_zarr_if_exists(CITY, 'images_siamese_valid_tt', path=DATA_DIR)
    utils.delete_zarr_if_exists(CITY, 'images_siamese_valid_t0', path=DATA_DIR)

if DATASET=='test' or DATASET=='all':
    utils.delete_zarr_if_exists(CITY, 'labels_siamese_test', path=DATA_DIR)
    utils.delete_zarr_if_exists(CITY, 'images_siamese_test_tt', path=DATA_DIR)
    utils.delete_zarr_if_exists(CITY, 'images_siamese_test_t0', path=DATA_DIR)
    
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

for j, pre_image_index in enumerate(PRE_IMAGE_INDEX):
    print(f'------ Using pre image #{j+1}..')
    if WINDOW:
        window = utils.center_window(images[pre_image_index], 
            (WINDOW_SIZE[0]*TILE_SIZE[0], WINDOW_SIZE[1]*TILE_SIZE[1]), TILE_SIZE,
                xoffset=XOFFSET, yoffset=YOFFSET)
        pre_image = utils.read_raster(images[pre_image_index], window=window)
    else:
        pre_image = utils.read_raster(images[pre_image_index])
    
    if HOG:
        pre_image = utils.get_hog(pre_image)
    
    pre_image = utils.tile_sequences(np.array([pre_image]), TILE_SIZE)
    
    for i in range(len(images)):
        if i not in PRE_IMAGE_INDEX:
            if WINDOW:
                window = utils.center_window(labels[i], (WINDOW_SIZE[0]*1, WINDOW_SIZE[1]*1), TILE_SIZE,
                    xoffset=XOFFSET, yoffset=YOFFSET)
                label = np.array(utils.read_raster(labels[i], window=window))
            else:
                label = np.array(utils.read_raster(labels[i]))
            label = label.flatten()
            exclude = np.where(label==-1.0)
            label = np.delete(label, exclude)
            samples_valid = np.delete(samples.flatten(), exclude)
            _, label_train, label_test, label_valid = utils.sample_split(label, samples_valid )

            if DATASET=='train' or DATASET=='all':
                utils.save_zarr(np.equal(label_train, 3), CITY, 'labels_siamese_train', path=DATA_DIR)
            if DATASET=='validate' or DATASET=='all':
                utils.save_zarr(np.equal(label_valid, 3), CITY, 'labels_siamese_valid', path=DATA_DIR)
            if DATASET=='test' or DATASET=='all':
                utils.save_zarr(np.equal(label_test, 3), CITY, 'labels_siamese_test', path=DATA_DIR)

            
            if WINDOW:
                window = utils.center_window(images[i], (WINDOW_SIZE[0]*TILE_SIZE[0], WINDOW_SIZE[1]*TILE_SIZE[1]), TILE_SIZE, xoffset=XOFFSET, yoffset=YOFFSET)
                image = np.array(utils.read_raster(images[i], window=window))
            else:
                image = np.array(utils.read_raster(images[i]))

            if HOG:
                image = utils.get_hog(image)

            image = utils.tile_sequences(np.array([image]), TILE_SIZE)
            image = np.delete(image, exclude, 0)
            _, image_train, image_test, image_valid = utils.sample_split(image, samples_valid)
            if DATASET=='train' or DATASET=='all':
                utils.save_zarr(utils.flatten_image(image_train), CITY, 'images_siamese_train_tt', path=DATA_DIR)
            if DATASET=='validate' or DATASET=='all':
                utils.save_zarr(utils.flatten_image(image_valid), CITY, 'images_siamese_valid_tt', path=DATA_DIR)
            if DATASET=='test' or DATASET=='all':
                utils.save_zarr(utils.flatten_image(image_test), CITY, 'images_siamese_test_tt', path=DATA_DIR)
 
            pre_image_v = np.delete(pre_image, exclude, 0)
            _, pre_image_train, pre_image_test, pre_image_valid = utils.sample_split(pre_image_v, samples_valid)
            if DATASET=='train' or DATASET=='all':
                utils.save_zarr(utils.flatten_image(pre_image_train), CITY, 'images_siamese_train_t0', path=DATA_DIR)
            if DATASET=='validate' or DATASET=='all':
                utils.save_zarr(utils.flatten_image(pre_image_valid), CITY, 'images_siamese_valid_t0', path=DATA_DIR)
            if DATASET=='test' or DATASET=='all':
                utils.save_zarr(utils.flatten_image(pre_image_test), CITY, 'images_siamese_test_t0', path=DATA_DIR)
            print(f'--------- Image {i+1 - len(PRE_IMAGE_INDEX)} of {len(images) - len(PRE_IMAGE_INDEX)} done..')

if DATASET=='train' or DATASET=='all':
    # Generate a balanced (upsampled) dataset and shuffle it..
    utils.delete_zarr_if_exists(CITY, 'labels_siamese_train_balanced')
    utils.delete_zarr_if_exists(CITY, 'images_siamese_train_t0_balanced')
    utils.delete_zarr_if_exists(CITY, 'images_siamese_train_tt_balanced')
    if BALANCE:
        print('--- Generate a balanced (upsampled) dataset..')
        utils.balance_snn(CITY)
    print('--- Shuffle dataset..')
    utils.shuffle_snn(CITY, TILE_SIZE, (100,750))
