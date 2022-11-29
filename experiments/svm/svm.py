import os 
import rasterio
import numpy as np 
from sklearn import svm
import re
import time 
import random

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
    


    # print(image.shape)

    
    label = read_raster(label, 1)
    label = np.squeeze(label.flatten())

    # Remove uncertain
    unc = np.where(label == -1)
    # label = label[list(set(np.arange(len(label))) - set(unc[0]))]
    # label[np.where(label != 3)] = 0.0
    # label[np.where(label == 3)] = 1.0

    # print(label.shape)


    # Flatten and reshape
    image = read_raster(image, 1)

    image = tile_sequences(np.array([image]))
    image = np.squeeze(image)
    # image = image[list(set(np.arange(len(image))) - set(unc[0]))]
    n, r, c = image.shape
    image = image.reshape(n, r*c)

    # print("DEBUG", images[i] )
    # print(image.shape)
    # print(label.shape)
    # print(samples.shape)
    # print("END DEBUG")



    # Sample split
    # _, image_tr, image_va, image_te = sample_split(image, samples.flatten())
    # _, label_tr, label_va, label_te = sample_split(label, samples.flatten())
    image_tr, image_va, image_te = sample_split(image, samples.flatten()) # for smaller samples there is no noanalysis class
    label_tr, label_va, label_te = sample_split(label, samples.flatten())



    # print("Appending:", image_tr.shape[0], image_va.shape[0], image_te.shape[0])
    images_tr = np.append(images_tr, image_tr, axis=0)
    images_va = np.append(images_va, image_va, axis=0)
    images_te = np.append(images_te, image_te, axis=0)

    labels_tr = np.append(labels_tr, label_tr, axis=0)
    labels_va = np.append(labels_va, label_va, axis=0)
    labels_te = np.append(labels_te, label_te, axis=0)

    # print("Append complete:", image_tr.shape, images_tr.shape)


# print("Before removed uncertain:",  images_tr.shape)
# print(np.unique(labels_tr, return_counts=True))

def remove_unc(images, labels):
    exclude = list(np.where(labels == -1))
    # certain = np.where(labels != -1)
    # images = images[certain]
    # labels = labels[certain]
    labels[labels != 3.0] = 0.0
    labels[labels == 3.0] = 1.0
    return np.delete(images, exclude, axis=0), np.delete(labels, exclude, axis=0)


images_tr, labels_tr = remove_unc(images_tr, labels_tr)
images_te, labels_te = remove_unc(images_te, labels_te)
images_va, labels_va = remove_unc(images_va, labels_va)


# labels_tr = labels_tr[list(np.where(labels_tr) != -1.0)]
# images_tr = images_tr[list(np.where(labels_tr) != -1.0)]


# print(np.unique(labels_tr, return_counts=True))


# Balance
print("Balancing..")
pos = np.where(labels_tr == 1.0)[0]
n_pos, n_neg = len(pos), len(labels_tr)-len(pos)
n_bal = n_neg - n_pos
# print(n_bal)
bal = random.choices(pos, k = n_bal)
# print(bal)
labels_tr = np.append(labels_tr, labels_tr[bal])

# print(images_tr.shape, images_tr[0:2].shape)
images_tr = np.append(images_tr, images_tr[bal], axis=0)



# # Shuffle
shuffled = np.arange(0, labels_tr.shape[0])
random.shuffle(shuffled)
labels_tr = labels_tr[shuffled]
images_tr = images_tr[shuffled]

# # PCA

# print(images_tr.shape)
# # # # # fit the model
clf = svm.SVC(gamma="auto", verbose=1)
clf.fit(images_tr, labels_tr)


yhat = clf.predict(images_te)
y = labels_te

print(y)