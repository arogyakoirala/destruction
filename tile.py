import os 
import rasterio
import numpy as np 
import re
import time 
import random
import zarr
import shutil
import matplotlib.pyplot as plt
import gc

CITY = 'aleppo'
DATA_DIR = "../data"
TILE_SIZE = (128,128)
PRE_IMAGE_INDEX=[0,1]
SUFFIX = "im" 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--city", help="City")
args = parser.parse_args()

if args.city:
    CITY = args.city



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

def read_raster(source:str, band:int=None, window=None, dtype:str='uint8', profile:bool=False) -> np.ndarray:
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

def sample_split(images:np.ndarray, samples:dict) -> list:
    '''Splits the data structure into multiple samples'''
    samples = [images[samples == value, ...] for value in np.unique(samples)]
    return samples



def read_zarr(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    return zarr.open(path)

def save_zarr(data, city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    if not os.path.exists(path):
        zarr.save(path, data)        
    else:
        za = zarr.open(path, mode='a')
        za.append(data)


def delete_zarr_if_exists(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    if os.path.exists(path):
        shutil.rmtree(path)


# images  = search_data(pattern(city=CITY, type='image'), directory=DATA_DIR)
pre_images  = search_data(pattern='^.*tif', directory=f'{DATA_DIR}/{CITY}/images/pre')
post_images  = search_data(pattern='^.*tif', directory=f'{DATA_DIR}/{CITY}/images/post')
labels  = search_data(pattern(city=CITY, type='label'), directory=DATA_DIR)
samples = read_raster(f'{DATA_DIR}/{CITY}/others/{CITY}_samples.tif')





image_dates = sorted([el.split("image_")[1].split('.tif')[0] for el in [*post_images]])
label_dates = sorted([el.split("label_")[1].split('.tif')[0] for el in labels])

# # print(image_dates)
# # print(label_dates)


remove = []
for la in label_dates:
    if la.replace("-", "_") not in image_dates:
        # print(la, "not in image_dates" )
        remove.append(label_dates.index(la))

_ = []
_labels = []
for i, dt in enumerate(label_dates):
    if i not in remove:
        _.append(dt)
        _labels.append(labels[i])


label_dates = sorted(_)
labels = sorted(_labels)
# print(len(post_images), len(labels))
# print(image_dates, label_dates)


suffixes = ["im_tr_pre", "im_va_pre", "im_te_pre", "im_tr_post", "im_va_post", "im_te_post",  "la_tr",  "la_va",  "la_te"]
for s in suffixes:
    delete_zarr_if_exists(CITY, s, DATA_DIR)

f = open(f"{DATA_DIR}/{CITY}/others/metadata.txt", "a")

print(f'{len(post_images)} post images; {len(pre_images)} pre images')
f.write("######## Tiling Step\n\n")
f.write(f'{len(post_images)} post images; {len(pre_images)} pre images\n')

# # empty = np.empty((0, TILE_SIZE[0]*TILE_SIZE[1]))
# # images_tr = images_va = images_te = empty
# # labels_tr = labels_va = labels_te = np.empty((0,))

for j, pre_image in enumerate(pre_images):
    print(f"Reading pre image.. {pre_images[j]}")
    pre_image = read_raster(pre_images[j])
    print(f"Tiling pre image.. {pre_images[j]}")
    pre_image = tile_sequences(np.array([pre_image]), TILE_SIZE)


    for i in range(len(post_images)):
        label = labels[i]
        label = read_raster(label, 1)
        label = np.squeeze(label.flatten())
        unc = np.where(label == -1)
        label = np.delete(label, unc, 0)

        image = post_images[i]
        print(f"Reading post image.. {post_images[i]}")
        image = read_raster(image)
        image = tile_sequences(np.array([image]))
        image = np.squeeze(image)
        image = np.delete(image, unc, 0)

        _pre_image = np.delete(pre_image, unc, 0)
        
        samples_min_unc = np.delete(samples.flatten(), unc)
        _, pre_image_tr, pre_image_va, pre_image_te = sample_split(_pre_image, samples_min_unc)
        _, image_tr, image_va, image_te = sample_split(image, samples_min_unc) # for smaller samples there is no noanalysis class
        _, label_tr, label_va, label_te = sample_split(label, samples_min_unc)  


        # image_tr, image_va, image_te = sample_split(image, samples_min_unc) # for smaller samples there is no noanalysis class
        # label_tr, label_va, label_te = sample_split(label, samples_min_unc)  
        # pre_image_tr, pre_image_va, pre_image_te = sample_split(_pre_image, samples_min_unc)

        # image_tr = image_tr.reshape(*image_tr.shape)
        pre_image_tr = np.squeeze(pre_image_tr)
        save_zarr(pre_image_tr, CITY, 'im_tr_pre', path=DATA_DIR)
        save_zarr(image_tr, CITY, 'im_tr_post', path=DATA_DIR)
        save_zarr(label_tr, CITY, 'la_tr', path=DATA_DIR)

        pre_image_va = np.squeeze(pre_image_va)
        save_zarr(pre_image_va, CITY, 'im_va_pre', path=DATA_DIR)
        save_zarr(image_va, CITY, 'im_va_post', path=DATA_DIR)
        save_zarr(label_va, CITY, 'la_va', path=DATA_DIR)

        pre_image_te = np.squeeze(pre_image_te)
        save_zarr(pre_image_te, CITY, 'im_te_pre', path=DATA_DIR)
        save_zarr(image_te, CITY, 'im_te_post', path=DATA_DIR)
        save_zarr(label_te, CITY, 'la_te', path=DATA_DIR)
        

        # save_zarr(image_tr, CITY, 'im_snn_tr_tt', path=DATA_DIR)
        # save_zarr(pre_image_tr, CITY, 'im_snn_tr_t0', path=DATA_DIR)
                


print("Sanity Check 1: Training")

tr_pre = read_zarr(CITY, "im_tr_pre", DATA_DIR)
tr_post = read_zarr(CITY, "im_tr_post", DATA_DIR)
la_tr = read_zarr(CITY, "la_tr", DATA_DIR)
index = random.randint(0,tr_pre.shape[0] - 10)


# print("Pre", tr_pre.shape)
# print("Post",tr_post.shape)
# print(la_tr.shape)

fig, ax = plt.subplots(2,5,dpi=200, figsize=(25,10))
ax = ax.flatten()
for i, image in enumerate(tr_pre[index:index+5]):
    ax[i].imshow(image)
for i, image in enumerate(tr_post[index:index+5]):
    ax[i+5].imshow(image)
plt.suptitle("Training set (sample images; top=pre, bottom=post)")
plt.savefig(f"{DATA_DIR}/{CITY}/others/tr_samples.png")



print("Sanity Check 2: Testing")
te_pre = read_zarr(CITY, "im_te_pre", DATA_DIR)
te_post = read_zarr(CITY, "im_te_post", DATA_DIR)
la_te = read_zarr(CITY, "la_te", DATA_DIR)
index = random.randint(0,te_pre.shape[0] - 10)


# print("Pre", te_pre.shape)
# print("Post",te_post.shape)
# print(la_te.shape)

fig, ax = plt.subplots(2,5,dpi=200, figsize=(25,10))
ax = ax.flatten()
for i, image in enumerate(te_pre[index:index+5]):
    ax[i].imshow(image)
for i, image in enumerate(te_post[index:index+5]):
    ax[i+5].imshow(image)
plt.suptitle("Test set (sample images; top=pre, bottom=post)")
plt.savefig(f"{DATA_DIR}/{CITY}/others/te_samples.png")

print("Sanity Check 1: Validation")
va_pre = read_zarr(CITY, "im_va_pre", DATA_DIR)
va_post = read_zarr(CITY, "im_va_post", DATA_DIR)
la_va = read_zarr(CITY, "la_va", DATA_DIR)
index = random.randint(0,va_pre.shape[0] - 10)

# print("Pre", va_pre.shape)
# print("Post",va_post.shape)
# print(la_va.shape)

fig, ax = plt.subplots(2,5,dpi=200, figsize=(25,10))
ax = ax.flatten()
for i, image in enumerate(va_pre[index:index+5]):
    ax[i].imshow(image)
for i, image in enumerate(va_post[index:index+5]):
    ax[i+5].imshow(image)
plt.suptitle("Validation set (sample images; top=pre, bottom=post)")
plt.savefig(f"{DATA_DIR}/{CITY}/others/va_samples.png")

f.write(f"Training set: {tr_pre.shape[0]} observations\n")
f.write(f"Validation set: {va_pre.shape[0]} observations\n")
f.write(f"Test set: {te_pre.shape[0]} observations\n\n")
f.close()
