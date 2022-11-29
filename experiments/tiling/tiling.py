import os 
import rasterio
import numpy as np 
import re
import time 
import random
import zarr

CITY = 'aleppo_cropped'
DATA_DIR = "../../../data"
TILE_SIZE = (128,128)
# image = '../hogs/image_2015_04_26.tif'
# label = '../../../data/aleppo_cropped/labels/label_2015-04-26.tif'

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

def sample_split(images:np.ndarray, samples:dict) -> list:
    '''Splits the data structure into multiple samples'''
    samples = [images[samples == value, ...] for value in np.unique(samples)]
    return samples



def save_zarr(data, city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    if not os.path.exists(path):
        zarr.save(path, data)        
    else:
        za = zarr.open(path, mode='a')
        za.append(data)

images  = search_data(pattern(city=CITY, type='image'), directory=DATA_DIR)
labels  = search_data(pattern(city=CITY, type='label'), directory=DATA_DIR)
samples = read_raster(f'{DATA_DIR}/{CITY}/others/{CITY}_samples.tif')



image_dates = sorted([el.split("image_")[1].split('.tif')[0] for el in images])
label_dates = sorted([el.split("label_")[1].split('.tif')[0] for el in labels])

# print(image_dates)
# print(label_dates)

remove = []
for la in label_dates:
    if la.replace("-", "_") not in image_dates:
        print(la, "not in image_dates" )
        remove.append(label_dates.index(la))

_ = []
for i, dt in enumerate(label_dates):
    if i not in remove:
        _.append(dt)

label_dates = sorted(_)
# print(len(image_dates), len(label_dates))


empty = np.empty((0, TILE_SIZE[0]*TILE_SIZE[1]))
images_tr = images_va = images_te = empty
labels_tr = labels_va = labels_te = np.empty((0,))

for i, image in enumerate(images):
    image = images[i]
    label = labels[i]
    
    label = read_raster(label, 1)
    label = np.squeeze(label.flatten())

    unc = np.where(label == -1)


    # Flatten and reshape
    image = read_raster(image)

    image = tile_sequences(np.array([image]))
    image = np.squeeze(image)
    print(image.shape)
    n, r, c, b = image.shape
    # image = image.reshape(n, r*c)

    # Sample split
    # _, image_tr, image_va, image_te = sample_split(image, samples.flatten())
    # _, label_tr, label_va, label_te = sample_split(label, samples.flatten())
    image_tr, image_va, image_te = sample_split(image, samples.flatten()) # for smaller samples there is no noanalysis class
    label_tr, label_va, label_te = sample_split(label, samples.flatten())
    # print(image_tr.shape)  
    # print(image_va.shape)  
    # print(image_te.shape)  


    save_zarr(city=CITY, data=image_tr, suffix="im_tr", path=DATA_DIR)
    save_zarr(city=CITY, data=image_va, suffix="im_va", path=DATA_DIR)
    save_zarr(city=CITY, data=image_te, suffix="im_te", path=DATA_DIR)

    save_zarr(city=CITY, data=label_tr, suffix="la_tr", path=DATA_DIR)
    save_zarr(city=CITY, data=label_va, suffix="la_va", path=DATA_DIR)
    save_zarr(city=CITY, data=label_te, suffix="la_te", path=DATA_DIR)

