import zarr
import numpy as np
import os
import shutil
import random
import math

# SUFFIX = 'im'
CITY = 'aleppo_cropped'
DATA_DIR = "../../../data"
BLOCK_SIZE = 10000

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

def delete_zarr_if_exists(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    if os.path.exists(path):
        shutil.rmtree(path)

def shuffle(old="tr", new="tr_sfl", delete_old=False):
    images_pre = read_zarr(city=CITY, suffix=f"im_{old}_pre", path=DATA_DIR)
    images_post= read_zarr(city=CITY, suffix=f"im_{old}_post", path=DATA_DIR)
    labels = read_zarr(city=CITY, suffix=f"la_{old}", path=DATA_DIR)
    print("Shape before shuffle", images_pre.shape)
    print("Shape before shuffle", images_post.shape)

    n = images_pre.shape[0]
    blocks = make_tuple_pair(n, BLOCK_SIZE)
    np.random.shuffle(blocks)
    print(blocks)

    for bl in blocks:
        
        im_pre = images_pre[bl[0]: bl[1]]
        im_post = images_post[bl[0]: bl[1]]
        la = labels[bl[0]: bl[1]]

        r = np.arange(0, bl[1] - bl[0])
        np.random.shuffle(r)
        im_pre = im_pre[r]
        im_post = im_post[r]
        la = la[r]
        save_zarr(data=im_pre, city=CITY, suffix=f"im_{new}_pre", path=DATA_DIR)
        save_zarr(data=im_post, city=CITY, suffix=f"im_{new}_post", path=DATA_DIR)
        save_zarr(data=la[r], city=CITY, suffix=f"la_{new}", path=DATA_DIR)

    if delete_old == True:
        delete_zarr_if_exists(CITY, f"im_{old}_pre", DATA_DIR)
        delete_zarr_if_exists(CITY, f"im_{old}_post", DATA_DIR)
        delete_zarr_if_exists(CITY, f"la_{old}", DATA_DIR)


shuffle(old="tr", new="tr_sfl", delete_old=True)
shuffle(old="tr_sfl", new="tr", delete_old=True)
print("Sanity check (size)", read_zarr(city=CITY, suffix=f"im_tr_pre", path=DATA_DIR).shape)
print("Sanity check (size)", read_zarr(city=CITY, suffix=f"im_tr_pre", path=DATA_DIR).shape)
print("Sanity check (size)", read_zarr(city=CITY, suffix=f"la_tr", path=DATA_DIR).shape)
print("Sanity check (shuffling)", read_zarr(city=CITY, suffix=f"la_tr", path=DATA_DIR)[-10:])

