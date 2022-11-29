import zarr
import numpy as np
from skimage.feature import hog
import os


CITY = "aleppo_cropped"
TILE_SIZE = (128,128)
DATA_DIR = "../../../data"


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


def get_hog(image):
    image = np.float32(image)
    _, image = hog(image, orientations=9, pixels_per_cell=(2,2),
                	cells_per_block=(4, 4), channel_axis=2, visualize=True)
    return _, image    

def get_hog_patches(images):
    n, h,w,b = images.shape
    hogs = np.empty((0, h,w))
    # encodings = np.empty(0, h,w,b)

    for im in images:
        _, image = get_hog(im)
        image = image.reshape(1, h,w)
        hogs = np.append(hogs, image, axis=0) 
        # print(_.shape)






images_tr = read_zarr('aleppo_cropped', 'im_tr', DATA_DIR)[:]
images_tr_hog = get_hog_patches(images_tr)
save_zarr(city=CITY, data=images_tr_hog, suffix="hog_tr", path=DATA_DIR)


images_va = read_zarr('aleppo_cropped', 'im_va', DATA_DIR)[:]
images_va_hog = get_hog_patches(images_va)
save_zarr(city=CITY, data=images_va_hog, suffix="hog_va", path=DATA_DIR)


images_te = read_zarr('aleppo_cropped', 'im_te', DATA_DIR)[:]
images_te_hog = get_hog_patches(images_te)
save_zarr(city=CITY, data=images_te_hog, suffix="hog_te", path=DATA_DIR)


# hv(images_tr)