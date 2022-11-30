import zarr
import numpy as np
from skimage.feature import hog
import os
import matplotlib.pyplot as plt
import shutil
import gc

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
        if i%1000==0:
            print(i)
        _, image = get_hog(im)
        if encodings is None:
            encodings = np.empty((0, _.shape[0]))
        
        image = image.reshape(1, h,w)
        hogs = np.append(hogs, image, axis=0) 
        encodings = np.append(encodings, _.reshape(1, _.shape[0]), axis=0)
        gc.collect()
    return hogs, encodings

images_tr = read_zarr('aleppo_cropped', 'im_tr', DATA_DIR)[:]
print(images_tr.shape)
images_tr_hog, images_tr_hoge = get_hog_patches(images_tr)
save_zarr(city=CITY, data=images_tr_hog, suffix="hog_tr", path=DATA_DIR)
save_zarr(city=CITY, data=images_tr_hoge, suffix="hoge_tr", path=DATA_DIR)

images_va = read_zarr('aleppo_cropped', 'im_va', DATA_DIR)[:]
images_va_hog, images_va_hoge = get_hog_patches(images_va)
save_zarr(city=CITY, data=images_va_hog, suffix="hog_va", path=DATA_DIR)
save_zarr(city=CITY, data=images_va_hoge, suffix="hoge_va", path=DATA_DIR)

images_te = read_zarr('aleppo_cropped', 'im_te', DATA_DIR)[:]
images_te_hog, images_te_hoge = get_hog_patches(images_te)
save_zarr(city=CITY, data=images_te_hog, suffix="hog_te", path=DATA_DIR)
save_zarr(city=CITY, data=images_te_hoge, suffix="hoge_te", path=DATA_DIR)



# Show the first 10 images
# fig, ax = plt.subplots(2,5,figsize=(8,3), dpi=300)
# axes = ax.flatten()
# for i in range(10):
#     axes[i].imshow(images_tr[i, :, :, 1].reshape(128,128), cmap="gray")
# for ax in axes:
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.title("Original images")
# plt.tight_layout()
# plt.savefig(f"image_samples.png")

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