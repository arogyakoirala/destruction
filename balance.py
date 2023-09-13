import zarr
import numpy as np
import os
import shutil
import random

# SUFFIX = 'im_tr'
CITY = 'aleppo'
DATA_DIR = "../data"
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
        
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--city", help="City")
parser.add_argument("--data_dir", help="Data dir")
args = parser.parse_args()

if args.city:
    CITY = args.city

if args.data_dir:
    DATA_DIR = args.data_dir

images_pre = read_zarr(CITY, 'im_tr_pre', DATA_DIR)
images_post = read_zarr(CITY, 'im_tr_post', DATA_DIR)
labels = read_zarr(CITY, 'la_tr', DATA_DIR)

def make_tuple_pair(n, step_size):
    if step_size > n:
        return [(0,n)]
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

blocks = make_tuple_pair(labels.shape[0], BLOCK_SIZE)

pos = []
for i, bl in enumerate(blocks):
    im_pre = images_pre[bl[0]: bl[1]]
    im_post = images_post[bl[0]: bl[1]]
    la = labels[bl[0]: bl[1]]

    p = list(*np.where(la==1))
    p = [(i * BLOCK_SIZE) + l for l in p]
    for _ in p:
        pos.append(_)
    # print(pos)
    # for p in pos:
    #     print(labels[p])
        
pos = sorted(pos)
neg = labels.shape[0] - len(pos)
print(pos, neg)
add = random.choices(pos, k=(neg - len(pos)))
add = sorted(add)

for i, bl in enumerate(blocks):
    im_pre = images_pre[bl[0]: bl[1]]
    im_post = images_post[bl[0]: bl[1]]
    la = labels[bl[0]: bl[1]]

    ind = [j - (i*BLOCK_SIZE) for j in add if j >= bl[0] and j < bl[1]]
    if len(ind) > 0:
        # print(im[ind].shape)
        save_zarr(im_pre[ind], CITY, "im_tr_pre", DATA_DIR)
        save_zarr(im_post[ind], CITY, "im_tr_post", DATA_DIR)
        save_zarr(la[ind], CITY, "la_tr", DATA_DIR)

print(f"There were {labels.shape[0]} total. {neg} negative; {len(pos)} positive. Added {len(add)} positive samples")
labels = read_zarr(CITY, 'la_tr', DATA_DIR)
print(f"New shape: {labels.shape}")

f = open(f"{DATA_DIR}/{CITY}/others/metadata.txt", "a")

f.write("\n\n######## Balancing Step\n\n")
f.write(f"There were {labels.shape[0]} total. {neg} negative; {len(pos)} positive. Added {len(add)} positive samples\n")

tr_pre = read_zarr(CITY, "im_tr_pre", DATA_DIR)
va_pre = read_zarr(CITY, "im_va_pre", DATA_DIR)
te_pre = read_zarr(CITY, "im_te_pre", DATA_DIR)

f.write(f"Training set: {tr_pre.shape[0]} observations\n")
f.write(f"Validation set: {va_pre.shape[0]} observations\n")
f.write(f"Test set: {te_pre.shape[0]} observations\n\n")
f.close()
