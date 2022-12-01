import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--suffix", help="Suffix")
args = parser.parse_args()

SUFFIX = 'hog'
CITY = 'aleppo_cropped'
DATA_DIR = "../../../data"
PERC_VARIANCE = 0.80

if args.suffix:
    SUFFIX = args.suffix 

import os
import zarr
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil



def read_zarr(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    return zarr.open(path)

def standardize(data, mu=None, sigma=None, full = False):
    if mu is None:
        mu = np.mean(data, axis=0)
    if sigma is None:
        sigma = np.std(data, axis=0) + 0.00000000000001
    if not full:
        return ((data - mu) / sigma)
    return ((data - mu) / sigma), mu, sigma

def inverse_pca(pca_data, pca, remove_n):
    transformed = pca_data.copy()
    transformed[:, -remove_n:] = 0 
    return pca.inverse_transform(transformed)

def save_zarr(data, city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    if os.path.exists(path):
        shutil.rmtree(path)
    # if not os.path.exists(path):
    zarr.save(path, data)        
    # else:
        # za = zarr.open(path, mode='a')
        # za.append(data)
def delete_zarr_if_exists(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    if os.path.exists(path):
        shutil.rmtree(path)

suffixes = [f"pca_{SUFFIX}_tr_pre",f"pca_{SUFFIX}_va_pre",f"pca_{SUFFIX}_te_pre", f"pca_{SUFFIX}_tr_post",f"pca_{SUFFIX}_va_post",f"pca_{SUFFIX}_te_post"]
for s in suffixes:
    delete_zarr_if_exists(CITY, s, DATA_DIR)


images_pre = read_zarr(CITY, f"{SUFFIX}_tr_pre", DATA_DIR)[:]
n_pre = images_pre.shape
new_array = [tuple(row) for row in images_pre]
images_pre = np.unique(new_array, axis=0)
images_pre = images_pre.reshape(images_pre.shape[0], 128*128)
del new_array
print(f"images_pre shape: {images_pre.shape}")

images_tr_pre = read_zarr(CITY, f"{SUFFIX}_tr_pre", DATA_DIR)[:]
images_va_pre = read_zarr(CITY, f"{SUFFIX}_va_pre", DATA_DIR)[:]
images_te_pre = read_zarr(CITY, f"{SUFFIX}_te_pre", DATA_DIR)[:]
images_tr_post = read_zarr(CITY, f"{SUFFIX}_tr_post", DATA_DIR)[:]
images_va_post = read_zarr(CITY, f"{SUFFIX}_va_post", DATA_DIR)[:]
images_te_post = read_zarr(CITY, f"{SUFFIX}_te_post", DATA_DIR)[:]

images_tr_pre = images_tr_pre.reshape((images_tr_pre.shape[0], 128*128))
images_va_pre = images_va_pre.reshape((images_va_pre.shape[0], 128*128))
images_te_pre = images_te_pre.reshape((images_te_pre.shape[0], 128*128))
images_tr_post = images_tr_post.reshape((images_tr_post.shape[0], 128*128))
images_va_post = images_va_post.reshape((images_va_post.shape[0], 128*128))
images_te_post = images_te_post.reshape((images_te_post.shape[0], 128*128))

images_all = np.append(images_tr_post, images_va_post, axis=0)
images_all = np.append(images_all, images_te_post, axis=0)
images_all = np.append(images_all, images_pre, axis=0)



images_all = images_all.reshape((images_all.shape[0], 128*128))

# if SUFFIX != "hoge":
#     n, h, w = images_tr.shape
#     images = images_tr.reshape(n, h*w)


images_std, mu, sigma = standardize(images_all, full=True)
# print(images_std.shape)
k=min(len(images_std), 500)
pca = PCA(n_components=k)
pca.fit(images_std)
n, perc = min(enumerate(pca.explained_variance_ratio_.cumsum()), key=lambda x: abs(x[1]-PERC_VARIANCE))
print(f"{n} principal components explain ~{(100*perc).astype(int)}% of variance")

fig, ax = plt.subplots(1,1,figsize=(15,3), dpi=200)
sns.lineplot(x=range(1, k+1), y=pca.explained_variance_ratio_.cumsum(), ax=ax);
fig.suptitle('Total Variation Explained by Number of PCA Components', fontsize='xx-large')
ax.set_xlabel('Number of Components', fontsize='x-large')
ax.set_ylabel('% Variation Explained', fontsize='x-large')
plt.axvline(min(enumerate(pca.explained_variance_ratio_.cumsum()), key=lambda x: abs(x[1]-PERC_VARIANCE))[0], dashes=(1.0, 1.0))
plt.axhline(min(enumerate(pca.explained_variance_ratio_.cumsum()), key=lambda x: abs(x[1]-PERC_VARIANCE))[1], dashes=(1.0, 1.0))
plt.savefig(f"{SUFFIX}_pca_var_plot.png")



images_tr_pre = standardize(images_tr_pre, mu, sigma)
pcas = pca.transform(images_tr_pre)
pcas[:, -(k-n):] = 0
save_zarr(city=CITY, data=pcas, suffix=f"pca_{SUFFIX}_tr_pre", path=DATA_DIR)

images_va_pre = standardize(images_va_pre, mu, sigma)
pcas = pca.transform(images_va_pre)
pcas[:, -(k-n):] = 0
save_zarr(city=CITY, data=pcas, suffix=f"pca_{SUFFIX}_va_pre", path=DATA_DIR)

images_te_pre = standardize(images_te_pre, mu, sigma)
pcas = pca.transform(images_te_pre)
pcas[:, -(k-n):] = 0
save_zarr(city=CITY, data=pcas, suffix=f"pca_{SUFFIX}_te_pre", path=DATA_DIR)

images_tr_post = standardize(images_tr_post, mu, sigma)
pcas = pca.transform(images_tr_post)
pcas[:, -(k-n):] = 0
save_zarr(city=CITY, data=pcas, suffix=f"pca_{SUFFIX}_tr_post", path=DATA_DIR)

images_va_post = standardize(images_va_post, mu, sigma)
pcas = pca.transform(images_va_post)
pcas[:, -(k-n):] = 0
save_zarr(city=CITY, data=pcas, suffix=f"pca_{SUFFIX}_va_post", path=DATA_DIR)

images_te_post = standardize(images_te_post, mu, sigma)
pcas = pca.transform(images_te_post)
pcas[:, -(k-n):] = 0
save_zarr(city=CITY, data=pcas, suffix=f"pca_{SUFFIX}_te_post", path=DATA_DIR)

fig,ax = plt.subplots(2,5,sharex=True,sharey=True,figsize=(8,3), dpi=300)
axes = ax.flatten()
pcas = pca.transform(images_va_post[:5])
pcas[:, -(k-n):] = 0
for i, im in enumerate(pcas):
    result = pca.inverse_transform(im).reshape((128,128))
    axes[i].imshow(images_va_post[i].reshape(128,128), cmap="gray")
    axes[i+5].imshow(result, cmap="gray")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(f"Reconstr. with {n} principal components; explains ~{(100*perc).astype(int)}% of variance",fontsize=12)
plt.tight_layout()
plt.savefig(f"{SUFFIX}_reconstructed.png")
