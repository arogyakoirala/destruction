import zarr
import numpy as np
import os
import shutil
import random
import math
import matplotlib.pyplot as plt



# SUFFIX = 'im'
CITY = 'aleppo'
DATA_DIR = "../data"
BLOCK_SIZE = 2000

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--city", help="City")
args = parser.parse_args()

if args.city:
    CITY = args.city

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

def shuffle(old="tr", new="tr_sfl", delete_old=False, block_size = BLOCK_SIZE):
    images_pre = read_zarr(city=CITY, suffix=f"im_{old}_pre", path=DATA_DIR)
    images_post= read_zarr(city=CITY, suffix=f"im_{old}_post", path=DATA_DIR)
    labels = read_zarr(city=CITY, suffix=f"la_{old}", path=DATA_DIR)
    print("Shape before shuffle", images_pre.shape)
    print("Shape before shuffle", images_post.shape)

    n = images_pre.shape[0]
    blocks = make_tuple_pair(n, block_size)
    np.random.shuffle(blocks)

    for i, bl in enumerate(blocks):
        print(i+1, bl)
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


shuffle(old="tr", new="tr_sfl", delete_old=True, block_size=BLOCK_SIZE)
shuffle(old="tr_sfl", new="tr", delete_old=True, block_size=BLOCK_SIZE*5)



print("Sanity Check 1: Training")
tr_pre = read_zarr(CITY, "im_tr_pre", DATA_DIR)
tr_post = read_zarr(CITY, "im_tr_post", DATA_DIR)
la_tr = read_zarr(CITY, "la_tr", DATA_DIR)
index = random.randint(0,tr_pre.shape[0] - 10)


fig, ax = plt.subplots(2,5,dpi=200, figsize=(25,10))
ax = ax.flatten()
for i, image in enumerate(tr_pre[index:index+5]):
    ax[i].imshow(image)
for i, image in enumerate(tr_post[index:index+5]):
    ax[i+5].imshow(image)
plt.suptitle("Training set shuffled (sample images; top=pre, bottom=post)")
plt.savefig(f"{DATA_DIR}/{CITY}/others/tr_samples_sfl.png")

print("Sanity Check 2: Testing")
te_pre = read_zarr(CITY, "im_te_pre", DATA_DIR)

print("Sanity Check 1: Validation")
va_pre = read_zarr(CITY, "im_va_pre", DATA_DIR)

f = open(f"{DATA_DIR}/{CITY}/others/metadata.txt", "a")

f.write("\n\n######## Shuffling Step\n\n")
f.write(f"Training set: {tr_pre.shape[0]} observations\n")
f.write(f"Validation set: {va_pre.shape[0]} observations\n")
f.write(f"Test set: {te_pre.shape[0]} observations\n\n")
f.close()