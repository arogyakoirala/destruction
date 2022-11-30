import zarr
import numpy as np
from skimage.feature import hog
import os
import matplotlib.pyplot as plt
import shutil
import gc
import multiprocessing as mp


CITY = "aleppo_cropped"
TILE_SIZE = (128,128)
DATA_DIR = "../../../data"
N_CORES = 30
BLOCK_SIZE = 1000

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


def read_zarr(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    return zarr.open(path)

def save_zarr(data, city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    # if os.path.exists(path):
    #     shutil.rmtree(path)
    if not os.path.exists(path):
        zarr.save(path, data)        
    else:
        za = zarr.open(path, mode='a')
        za.append(data)


def get_hog(image):
    image = np.float32(image)
    _, image = hog(image, orientations=9, pixels_per_cell=(2,2),
                    cells_per_block=(2, 2), channel_axis=2, visualize=True)
    return _, image    

def get_hog_patches(images):
    n, h,w,b = images.shape
    hogs = np.empty((0, h,w))
    encodings = None

    for i, im in enumerate(images):
        if i%100==0:
            print(i)
        _, image = get_hog(im)
        if encodings is None:
            encodings = np.empty((0, _.shape[0]))
        
        image = image.reshape(1, h,w)
        hogs = np.append(hogs, image, axis=0) 
        encodings = np.append(encodings, _.reshape(1, _.shape[0]), axis=0)
        gc.collect()
    return hogs, encodings


def parallel_block_handler(blocks, images, suffix):
    for b in blocks:
        print(b)
        bl = images[b[0]: b[1]]
        hogs, encs = get_hog_patches(bl)
        save_zarr(city=CITY, data=hogs, suffix=suffix, path=DATA_DIR)


def save_patches_parallel(images, bls, suffix):
    print(bls)
    parent_chunks = np.array_split(blocks, N_CORES)
    pool = mp.Pool(processes=N_CORES)
    chunk_processes = [pool.apply_async(parallel_block_handler, args=(chunk, images, suffix)) for chunk in parent_chunks]
    chunk_results = [chunk.get() for chunk in chunk_processes]


if __name__ == '__main__':
    

    images_tr = read_zarr('aleppo_cropped', 'im_tr', DATA_DIR)
    blocks = make_tuple_pair(images_tr.shape[0], BLOCK_SIZE)
    save_patches_parallel(images_tr, blocks, 'hog_tr')


    images_va = read_zarr('aleppo_cropped', 'im_va', DATA_DIR)
    blocks = make_tuple_pair(images_va.shape[0], BLOCK_SIZE)
    save_patches_parallel(images_va, blocks, 'hog_va')

    images_te = read_zarr('aleppo_cropped', 'im_te', DATA_DIR)
    blocks = make_tuple_pair(images_te.shape[0], BLOCK_SIZE)
    save_patches_parallel(images_te, blocks, 'hog_te')

   

    images_tr_hog = read_zarr('aleppo_cropped', 'hog_tr', DATA_DIR)

    # Show the first 10 hog
    fig, ax = plt.subplots(2,5,figsize=(8,3), dpi=300)
    axes = ax.flatten()
    fig.suptitle("Images with their corresponding HOGs")
    for i in range(5):
        axes[i+5].imshow(images_tr_hog[i].reshape(128,128), cmap="gray")
        axes[i].imshow(images_tr[i, :, :, 1].reshape(128,128), cmap="gray")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"hogs.png")