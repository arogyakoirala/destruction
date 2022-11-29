import zarr
import numpy as np
from skimage.feature import hog


CITY = "aleppo_cropped"
TILE_SIZE = (128,128)
DATA_DIR = "../../../data"


def read_zarr(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    return zarr.open(path)

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
        print(_.shape)






images_tr = read_zarr('aleppo_cropped', 'im_tr', DATA_DIR)[:]
get_hog_patches(images_tr)
# hv(images_tr)