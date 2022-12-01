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
    print(data.shape, path)
    # if os.path.exists(path):
    #     shutil.rmtree(path)
    if not os.path.exists(path):
        zarr.save(path, data)        
    else:
        za = zarr.open(path, mode='a')
        za.append(data)

def delete_zarr_if_exists(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    if os.path.exists(path):
        shutil.rmtree(path)



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
        if i%500==0:
            print(i)
        _, image = get_hog(im)
        if encodings is None:
            encodings = np.empty((0, _.shape[0]))
        
        image = image.reshape(1, h,w)
        hogs = np.append(hogs, image, axis=0) 
        encodings = np.append(encodings, _.reshape(1, _.shape[0]), axis=0)
        gc.collect()
    return hogs, encodings


def parallel_block_handler(blocks, images_pre, images_post, suffix):
    for b in blocks:
        print(b)
        bl_pre = images_pre[b[0]: b[1]]
        bl_post = images_post[b[0]: b[1]]
        hogs_pre, encs = get_hog_patches(bl_pre)
        hogs_post, encs = get_hog_patches(bl_post)
        # print(hogs_post.shape)
        save_zarr(city=CITY, data=hogs_pre, suffix=f"{suffix}_pre", path=DATA_DIR)
        save_zarr(city=CITY, data=hogs_post, suffix=f"{suffix}_post", path=DATA_DIR)


def save_patches_parallel(images_pre, images_post, blocks, suffix):
    parent_chunks = np.array_split(blocks, N_CORES)
    pool = mp.Pool(processes=N_CORES)
    chunk_processes = [pool.apply_async(parallel_block_handler, args=(chunk, images_pre, images_post, suffix)) for chunk in parent_chunks]
    chunk_results = [chunk.get() for chunk in chunk_processes]


if __name__ == '__main__':
    
    suffixes = ["hog_tr_pre", "hog_va_pre", "hog_te_pre", "hog_tr_post", "hog_va_post", "hog_te_post"]
    # suffixes = [f"{SUFFIX}_snn_tr", f"{SUFFIX}_snn_va", f"{SUFFIX}_snn_te", "la_snn_tr", "la_snn_va","la_snn_te"]
    for s in suffixes:
        delete_zarr_if_exists(CITY, s, DATA_DIR)

    images_tr_pre = read_zarr('aleppo_cropped', 'im_tr_pre', DATA_DIR)
    images_tr_post = read_zarr('aleppo_cropped', 'im_tr_post', DATA_DIR)
    blocks = make_tuple_pair(images_tr_pre.shape[0], BLOCK_SIZE)
    save_patches_parallel(images_tr_pre, images_tr_post, blocks, 'hog_tr')


    images_va_pre = read_zarr('aleppo_cropped', 'im_va_pre', DATA_DIR)
    images_va_post = read_zarr('aleppo_cropped', 'im_va_post', DATA_DIR)
    blocks = make_tuple_pair(images_va_pre.shape[0], BLOCK_SIZE)
    save_patches_parallel(images_va_pre, images_va_post, blocks, 'hog_va')

    images_te_pre = read_zarr('aleppo_cropped', 'im_te_pre', DATA_DIR)
    images_te_post = read_zarr('aleppo_cropped', 'im_te_post', DATA_DIR)
    blocks = make_tuple_pair(images_te_pre.shape[0], BLOCK_SIZE)
    save_patches_parallel(images_te_pre, images_te_post, blocks, 'hog_te')
   

    images_tr_hog = read_zarr('aleppo_cropped', 'hog_va_post', DATA_DIR)
    print(images_tr_hog)

    # Show the first 10 hog
    fig, ax = plt.subplots(2,5,figsize=(8,3), dpi=300)
    axes = ax.flatten()
    fig.suptitle("Images with their corresponding HOGs")
    for i in range(5):
        axes[i+5].imshow(images_tr_hog[i].reshape(128,128), cmap="gray")
        axes[i].imshow(images_va_post[i, :, :, 1].reshape(128,128), cmap="gray")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"hogs.png")