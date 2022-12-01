import zarr
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import gc
from scipy import signal
from scipy.ndimage import convolve

CITY = "aleppo_cropped"
TILE_SIZE = (128,128)
DATA_DIR = "../../../data"


def read_zarr(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    return zarr.open(path)

def save_zarr(data, city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    if os.path.exists(path):
        shutil.rmtree(path)
    # if not os.path.exists(path):
    zarr.save(path, data)        
    # else:
        # za = zarr.open(path, mode='a')
        # za.append(data)

# create a 2D Gaussian kernl
def create_2d_gaussian(size=9, std=1.5):
    gaussian_1d = signal.gaussian(size,std=std)
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
    gaussian_2d = gaussian_2d/(gaussian_2d.sum())
    return gaussian_2d

# normalize image between 0 and 1
def normalize_img(img):
    normalized = (img - img.min())/(img.max() - img.min())    
    return normalized


def gaussian_and_laplacian_stack(img, levels):
    gaussian = create_2d_gaussian(size=17, std=3)
    gaussian_stack = []
    img_gaussian = img.copy()
    for i in range(levels):
        if i == 0:
            gaussian_stack = [img_gaussian]
        else:
            gaussian_stack.append(convolve(gaussian_stack[-1], gaussian, mode='reflect'))
    
    laplacian_stack = []
    #
    # YOUR CODE HERE: create Laplcian stack and pack into the same type of data structure as gaussian_stack
    for k in range(0,levels-1):
        l1 = gaussian_stack[k]
        l2 = gaussian_stack[k+1]
        D  = l1 - l2
        laplacian_stack.append(D)
    laplacian_stack.append(gaussian_stack[levels-1])
        
    return (gaussian_stack, laplacian_stack)

def rgb2gray(rgb):
    # print(rgb)
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


# Creates an nd array of the first laplacian feature images of the input nd images array 
def get_laplacian(z_images,levels):
    """Returns a zarr file of the first gray image from the laplacian stage for all images in the
    original zarr file"""
    
    print("Greyscaling...")
    images_gray = rgb2gray(z_images)
    # print(f'The shape of {z_images} zarr file after rgb2gray conversion is {images_gray.shape}')
    lap_zarr = []
    print("Done greyscaling, building laplacian stack...")
    for i, img in enumerate(images_gray):
        if i%100 == 0:
            print(i)
        gs, ls = gaussian_and_laplacian_stack(img, levels)
        lap_zarr.append(ls[0])
        
    
    output = np.array(lap_zarr)
    len_output = len(output)
    output = np.reshape(output, (len_output, 128,128,1))
    
    return output


images_tr_pre = read_zarr('aleppo_cropped', 'im_tr_pre', DATA_DIR)[:]
images_tr_lap_pre = get_laplacian(images_tr_pre, 3)
images_tr_post = read_zarr('aleppo_cropped', 'im_tr_post', DATA_DIR)[:]
images_tr_lap_post = get_laplacian(images_tr_post, 3)
save_zarr(city=CITY, data=images_tr_lap_pre, suffix="lap_tr_pre", path=DATA_DIR)
save_zarr(city=CITY, data=images_tr_lap_post, suffix="lap_tr_post", path=DATA_DIR)

images_va_pre = read_zarr('aleppo_cropped', 'im_va_pre', DATA_DIR)[:]
images_va_lap_pre = get_laplacian(images_va_pre, 3)
images_va_post = read_zarr('aleppo_cropped', 'im_va_post', DATA_DIR)[:]
images_va_lap_post = get_laplacian(images_va_post, 3)
save_zarr(city=CITY, data=images_va_lap_pre, suffix="lap_va_pre", path=DATA_DIR)
save_zarr(city=CITY, data=images_va_lap_post, suffix="lap_va_post", path=DATA_DIR)

images_te_pre = read_zarr('aleppo_cropped', 'im_te_pre', DATA_DIR)[:]
images_te_lap_pre = get_laplacian(images_te_pre, 3)
images_te_post = read_zarr('aleppo_cropped', 'im_te_post', DATA_DIR)[:]
images_te_lap_post = get_laplacian(images_te_post, 3)
save_zarr(city=CITY, data=images_te_lap_pre, suffix="lap_te_pre", path=DATA_DIR)
save_zarr(city=CITY, data=images_te_lap_post, suffix="lap_te_post", path=DATA_DIR)


fig, ax = plt.subplots(2,5,figsize=(8,3), dpi=300)
axes = ax.flatten()
fig.suptitle("Images with their corresponding Level 1 Laplacians")
for i in range(5):
    axes[i+5].imshow(images_tr_lap_post[i].reshape(128,128), cmap="gray")
    axes[i].imshow(images_tr_post[i, :, :, 1].reshape(128,128), cmap="gray")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
plt.axis('off')
plt.tight_layout()
plt.savefig(f"laplacians.png")