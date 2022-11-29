# Modules
import os
import numpy as np
import re
import rasterio
from rasterio import features, windows
import geopandas
from matplotlib import pyplot
import zarr
import random
import gc
import shutil
from tensorflow.keras.utils import Sequence
import cv2
from skimage.feature import hog

# Raster Stuff
def tiled_profile(source:str, tile_size:tuple=(128,128,1)) -> dict:
    '''Computes raster profile for tiles'''
    raster  = rasterio.open(source)
    profile = raster.profile
    assert profile['width']  % tile_size[0] == 0, 'Invalid dimensions'
    assert profile['height'] % tile_size[1] == 0, 'Invalid dimensions'
    affine  = profile['transform']
    affine  = rasterio.Affine(affine[0] * tile_size[0], affine[1], affine[2], affine[3], affine[4] * tile_size[1], affine[5])
    profile.update(width=profile['width'] // tile_size[0], height=profile['height'] // tile_size[0], count=tile_size[2], transform=affine)
    return profile


def rasterise(source, profile, attribute:str=None, dtype:str='float32') -> np.ndarray:
    '''Tranforms vector data into raster'''
    if isinstance(source, str): 
        source = geopandas.read_file(source)
    if isinstance(profile, str): 
        profile = rasterio.open(profile).profile
    geometries = source['geometry']
    if attribute is not None:
        geometries = zip(geometries, source[attribute])
    image = features.rasterize(geometries, out_shape=(profile['height'], profile['width']), transform=profile['transform'])
    image = image.astype(dtype)
    return image

def display(image:np.ndarray, title:str='', cmap:str='gray', ax=None) -> None:
    '''Displays an image'''
    if ax==None:
        fig, ax = pyplot.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=20)
#     ax.legend(loc="upper left")
    ax.set_axis_off()
    pyplot.tight_layout()
    pyplot.show()

def display_multiple(images, labels=None, cmap='gray'):
    '''Displays an image'''
    fig, ax = pyplot.subplots(1, len(images), figsize=(30, 10))
    if len(images) > 1:
        ax = ax.flatten()
    else:
        ax = [ax]
    if labels == None:
        labels = []
        for i, image in enumerate(images):
            labels.append(i)
    
    
    for i, image in enumerate(images):
        ax[i].imshow(image, cmap=cmap)
        ax[i].set_title(labels[i], fontsize=20)
        ax[i].set_axis_off()
    pyplot.tight_layout()
    pyplot.show()
    
def write_raster(array:np.ndarray, profile, destination:str, nodata:int=None, dtype:str='float64') -> None:
    '''Writes a numpy array as a raster'''
    if array.ndim == 2:
        array = np.expand_dims(array, 2)
    array = array.transpose([2, 0, 1]).astype(dtype)
    bands, height, width = array.shape
    if isinstance(profile, str):
        profile = rasterio.open(profile).profile
    profile.update(driver='GTiff', dtype=dtype, count=bands, nodata=nodata)
    with rasterio.open(fp=destination, mode='w', **profile) as raster:
        raster.write(array)
        raster.close()
        
def read_raster(source:str, band:int=None, window=None, dtype:str='int') -> np.ndarray:
    '''Reads a raster as a numpy array'''
    raster = rasterio.open(source)
    if band is not None:
        image = raster.read(band, window=window)
        image = np.expand_dims(image, 0)
    else: 
        image = raster.read(window=window)
    image = image.transpose([1, 2, 0]).astype(dtype)
    return image

def extract(files:list, pattern:str='\d{4}-\d{2}-\d{2}') -> list:
    pattern = re.compile(pattern)
    match   = [pattern.search(file).group() for file in files]
    return match

def center_window(source:str, size:dict, tile_size:dict=(128,128), xoffset:int=0, yoffset:int=0):
    '''Computes the windows for the centre of a raster'''
    profile = rasterio.open(source).profile
    xoffset = xoffset * tile_size[0]
    yoffset = yoffset * tile_size[1]
    centre  = ((profile['width'] // 2), (profile['height'] // 2))
    window  = windows.Window.from_slices(
        (centre[0] - (size[0] // 2) + yoffset, centre[0] + size[0] // 2 + yoffset),
        (centre[1] - size[1] // 2 + xoffset, centre[1] + size[1] // 2 + xoffset)
    )
    return window

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

def flatten_image(image):
    p, d, h, w, b = image.shape
    return image.reshape((p,h,w,b))

# File IO Stuff
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

def read_zarr(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    return zarr.open(path)

def rename_zarr(city, old_suffix, new_suffix, path="../data"):
    old = f'{path}/{city}/others/{city}_{old_suffix}.zarr'
    new = f'{path}/{city}/others/{city}_{new_suffix}.zarr'
    os.rename(old, new)
    
    
# Data Stuff        
def balance(city, path="../data"):
    z_l = read_zarr(city, 'labels_conv_train', path=path)
    z_i = read_zarr(city, 'images_conv_train', path=path)

    path_l = f'{path}/{city}/others/{city}_labels_conv_train_balanced.zarr'
    path_i = f'{path}/{city}/others/{city}_images_conv_train_balanced.zarr'

    zarr.save(path_l, z_l)
    zarr.save(path_i, z_i)

    z_l_positives = np.where(np.squeeze(z_l) == 1)[0]
    z_l_negatives = np.where(np.squeeze(z_l) == 0)[0]
    sample_length = len(z_l_negatives) - len(z_l_positives)
    indices = random.choices(z_l_positives, k=sample_length)

    z_l_a = zarr.open(path_l, mode = 'a')
    z_i_a = zarr.open(path_i, mode = 'a')
    
    step_size = 5000
    for i, t in enumerate(make_tuple_pair(z_i.shape[0], step_size)):
        sub_indices = [num for num in indices if num >= t[0] and num < t[1]]
        sub_indices = list(map(lambda x: x-(i*step_size), sub_indices))
        to_add_l = z_l[t[0]:t[1]][sub_indices]
        to_add_i = z_i[t[0]:t[1]][sub_indices]
        z_l_a.append(to_add_l)
        z_i_a.append(to_add_i)

    gc.collect(generation=2)

    delete_zarr_if_exists(city, 'labels_conv_train', path=path)
    delete_zarr_if_exists(city, 'images_conv_train', path=path)

    rename_zarr(city, 'labels_conv_train_balanced', 'labels_conv_train', path=path)
    rename_zarr(city, 'images_conv_train_balanced', 'images_conv_train', path=path)
    
def shuffle(city, tile_size, batch_sizes, path="../data"):
    
    first, second = batch_sizes
    
    zarr_dir = f'{path}/{city}/others'
    path_l_b = f'{zarr_dir}/{city}_labels_conv_train.zarr'
    path_i_b = f'{zarr_dir}/{city}_images_conv_train.zarr'
    path_l_s = f'{zarr_dir}/{city}_labels_conv_train_shuffled.zarr'
    path_i_s = f'{zarr_dir}/{city}_images_conv_train_shuffled.zarr'

    z_l = zarr.open(path_l_b)
    z_i = zarr.open(path_i_b)
    n = z_l.shape[0]

    
    tuple_pair = make_tuple_pair(n, first)
    
    np.random.shuffle(tuple_pair)
    np.random.shuffle(tuple_pair)
    zarr.save(path_l_s, np.empty((0,1,1,1)))
    zarr.save(path_i_s, np.empty((0, *tile_size, 3)))

    z_l_s = zarr.open(path_l_s, mode='a')
    z_i_s = zarr.open(path_i_s, mode='a')
    print(f"------ Reordering array in batches of {first}. Total {len(tuple_pair)} sets..")
    
    for i, t in enumerate(tuple_pair):
        if i % 50 == 0 and i != 0:
            print(f"--------- Finished {i} sets")
        labels = z_l[t[0]:t[1]]
        images = z_i[t[0]:t[1]]
        z_l_s.append(labels)
        z_i_s.append(images)
        
    shutil.rmtree(path_l_b) 
    shutil.rmtree(path_i_b)

    del z_i_s, z_l_s, tuple_pair

    zarr.save(path_l_b, np.empty((0,1,1,1)))
    zarr.save(path_i_b, np.empty((0, *tile_size, 3)))

    z_l = zarr.open(path_l_b, mode='a')
    z_i = zarr.open(path_i_b, mode='a')
    z_l_s = zarr.open(path_l_s)
    z_i_s = zarr.open(path_i_s)
    tuple_pair = make_tuple_pair(n, second)
    print(f"------ Shuffling array in batches of {second}. Total {len(tuple_pair)} sets..")
    for i, t in enumerate(tuple_pair):
        if i % 15 == 0 and i != 0:
            print(f"--------- Finished {i} sets")
        shuffled = np.arange(0, t[1]-t[0])
        np.random.shuffle(shuffled)
        np.random.shuffle(shuffled)
        labels = z_l_s[t[0]:t[1]][shuffled]
        images = z_i_s[t[0]:t[1]][shuffled]
        z_l.append(labels)
        z_i.append(images)
    
    shutil.rmtree(path_l_s)
    shutil.rmtree(path_i_s)

# Misc Stuff

# Create tuple pairs for a given step size. used in balanced data generation
# Input = make_tuple_pair(16433, 5000)
# Output = [(0,5000), (5000, 10000), (10000,16433)] 
def make_tuple_pair(n, step_size):
    iters = n//step_size
    l = []
    for i in range(0, iters):
        if i == iters - 1:
            t = (i*step_size, n)
            l.append(t)
        else:
            t = (i*step_size, (i+1)*step_size)
            l.append(t)
    return l


def balance_snn(city, path="../data"):
    z_l = read_zarr(city, 'labels_siamese_train')
    z_i_t0 = read_zarr(city, 'images_siamese_train_t0')
    z_i_tt = read_zarr(city, 'images_siamese_train_tt')

    path_l = f'{path}/{city}/others/{city}_labels_siamese_train_balanced.zarr'
    path_i_t0 = f'{path}/{city}/others/{city}_images_siamese_train_t0_balanced.zarr'
    path_i_tt = f'{path}/{city}/others/{city}_images_siamese_train_tt_balanced.zarr'


    zarr.save(path_l, z_l)
    zarr.save(path_i_t0, z_i_t0)
    zarr.save(path_i_tt, z_i_tt)

    z_l_positives = np.where(np.squeeze(z_l) == 1)[0]
    z_l_negatives = np.where(np.squeeze(z_l) == 0)[0]
    sample_length = len(z_l_negatives) - len(z_l_positives)
    indices = random.choices(z_l_positives, k=sample_length)

    z_l_a = zarr.open(path_l, mode = 'a')
    z_i_t0_a = zarr.open(path_i_t0, mode = 'a')
    z_i_tt_a = zarr.open(path_i_tt, mode = 'a')
    
    step_size = 500
    for i, t in enumerate(make_tuple_pair(z_i_t0.shape[0], step_size)):
        sub_indices = [num for num in indices if num >= t[0] and num < t[1]]
        sub_indices = list(map(lambda x: x-(i*step_size), sub_indices))
        to_add_l = z_l[t[0]:t[1]][sub_indices]
        to_add_i_t0 = z_i_t0[t[0]:t[1]][sub_indices]
        to_add_i_tt = z_i_tt[t[0]:t[1]][sub_indices]
        z_l_a.append(to_add_l)
        z_i_t0_a.append(to_add_i_t0)
        z_i_tt_a.append(to_add_i_tt)

    gc.collect(generation=2)

    delete_zarr_if_exists(city, 'labels_siamese_train')
    delete_zarr_if_exists(city, 'images_siamese_train_t0')
    delete_zarr_if_exists(city, 'images_siamese_train_tt')


    rename_zarr(city, 'labels_siamese_train_balanced', 'labels_siamese_train')
    rename_zarr(city, 'images_siamese_train_t0_balanced', 'images_siamese_train_t0')
    rename_zarr(city, 'images_siamese_train_tt_balanced', 'images_siamese_train_tt')


def shuffle_snn(city, tile_size, batch_sizes, path="../data"):
    
    first, second = batch_sizes
    
    zarr_dir = f'{path}/{city}/others'
#     path_l = f'{zarr_dir}/{city}_labels_conv_train.zarr'
#     path_i = f'{zarr_dir}/{city}_images_conv_train.zarr'
    path_l_b = f'{zarr_dir}/{city}_labels_siamese_train.zarr'
    path_i_t0_b = f'{zarr_dir}/{city}_images_siamese_train_t0.zarr'
    path_i_tt_b = f'{zarr_dir}/{city}_images_siamese_train_tt.zarr'
    path_l_s = f'{zarr_dir}/{city}_labels_siamese_train_shuffled.zarr'
    path_i_t0_s = f'{zarr_dir}/{city}_images_siamese_train_t0_shuffled.zarr'
    path_i_tt_s = f'{zarr_dir}/{city}_images_siamese_train_tt_shuffled.zarr'

    z_l = zarr.open(path_l_b)
    z_i_t0 = zarr.open(path_i_t0_b)
    z_i_tt = zarr.open(path_i_tt_b)
    n = z_l.shape[0]

    tuple_pair = make_tuple_pair(n, first)
    np.random.shuffle(tuple_pair)
    np.random.shuffle(tuple_pair)
    zarr.save(path_l_s, np.empty((0)))
    zarr.save(path_i_t0_s, np.empty((0, *tile_size, 3)))
    zarr.save(path_i_tt_s, np.empty((0, *tile_size, 3)))

    z_l_s = zarr.open(path_l_s, mode='a')
    z_i_t0_s = zarr.open(path_i_t0_s, mode='a')
    z_i_tt_s = zarr.open(path_i_tt_s, mode='a')
    print(f"------ Reordering array in batches of {first}. Total {len(tuple_pair)} sets..")
    
    for i, t in enumerate(tuple_pair):
        if i % 50 == 0 and i != 0:
            print(f"--------- Finished {i} sets")
        labels = z_l[t[0]:t[1]]
        images_t0 = z_i_t0[t[0]:t[1]]
        images_tt = z_i_tt[t[0]:t[1]]
        z_l_s.append(labels)
        z_i_t0_s.append(images_t0)
        z_i_tt_s.append(images_tt)
        
    shutil.rmtree(path_l_b) 
    shutil.rmtree(path_i_t0_b)
    shutil.rmtree(path_i_tt_b)

    del z_i_t0_s, z_i_tt_s, z_l_s, tuple_pair

    zarr.save(path_l_b, np.empty((0)))
    zarr.save(path_i_t0_b, np.empty((0, *tile_size, 3)))
    zarr.save(path_i_tt_b, np.empty((0, *tile_size, 3)))

    z_l = zarr.open(path_l_b, mode='a')
    z_i_t0 = zarr.open(path_i_t0_b, mode='a')
    z_i_tt = zarr.open(path_i_tt_b, mode='a')
    z_l_s = zarr.open(path_l_s)
    z_i_t0_s = zarr.open(path_i_t0_s)
    z_i_tt_s = zarr.open(path_i_tt_s)
    tuple_pair = make_tuple_pair(n, second)
    print(f"------ Shuffling array in batches of {second}. Total {len(tuple_pair)} sets..")
    for i, t in enumerate(tuple_pair):
        if i % 15 == 0 and i != 0:
            print(f"--------- Finished {i} sets")
        print(t)
        shuffled = np.arange(0, second)
        np.random.shuffle(shuffled)
        np.random.shuffle(shuffled)
        labels = z_l_s[t[0]:t[1]][shuffled]
        images_t0 = z_i_t0_s[t[0]:t[1]][shuffled]
        images_tt = z_i_tt_s[t[0]:t[1]][shuffled]
        z_l.append(labels)
        z_i_t0.append(images_t0)
        z_i_tt.append(images_tt)
    
    shutil.rmtree(path_l_s)
    shutil.rmtree(path_i_t0_s)
    shutil.rmtree(path_i_tt_s)


    
class SiameseGenerator(Sequence):
    def __init__(self, images, labels, batch_size=32):
        self.images_t0 = images[0]
        self.images_tt = images[1]
        self.labels = labels
        self.batch_size = batch_size
        self.on_epoch_end()

   
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
        
    def make_tuple_pair(self, n, step_size):
        iters = n//step_size
        l = []
        for i in range(0, iters):
            if i == iters - 1:
                t = (i*step_size, n)
                l.append(t)
            else:
                t = (i*step_size, (i+1)*step_size)
                l.append(t)
        return l
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.tuple_pairs = make_tuple_pair(self.images_t0.shape[0], int(self.batch_size/4))
        np.random.shuffle(self.tuple_pairs)

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



class CNNGenerator(Sequence):
    def __init__(self, images, labels, batch_size=32):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.images)//self.batch_size
    
    def __getitem__(self, index):

        X = self.images[index*self.batch_size:(index+1)*self.batch_size]
        y = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        
        return self.augment(X), y.flatten()
    
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
        
        # Brightness
        alpha = random.choice(np.linspace(0.85, 1.4))
        X = X * alpha
        
        return X

def downsample(image, factor):
    image = cv2.GaussianBlur(image, (0, 0), 1, 1) # Blur with Gaussian kernel of width sigma=1
    image = cv2.resize(image, (0, 0), fx=1.0/factor, fy=1.0/factor, 
                           interpolation=cv2.INTER_CUBIC) # Downsample image by factor
    return image

def get_hog(image):
    image = np.float32(image)
    _, image = hog(image, orientations=9, pixels_per_cell=(2,2),
                	cells_per_block=(8, 8), channel_axis=2, visualize=True)
    return image        