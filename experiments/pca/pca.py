import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--suffix", "Suffix")
args = parser.parse_args()

SUFFIX = 'hog'
CITY = 'aleppo_cropped'
DATA_DIR = "../../../data"
PERC_VARIANCE = 0.95

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



images_tr = read_zarr(CITY, f"{SUFFIX}_tr", DATA_DIR)[:]
images_va = read_zarr(CITY, f"{SUFFIX}_va", DATA_DIR)[:]
images_te = read_zarr(CITY, f"{SUFFIX}_te", DATA_DIR)[:]

images_tr = images_tr.reshape((images_tr.shape[0], 128*128))
images_va = images_va.reshape((images_va.shape[0], 128*128))
images_te = images_te.reshape((images_te.shape[0], 128*128))

images_all = np.append(images_tr, images_va, axis=0)
images_all = np.append(images_all, images_te, axis=0)



images_all = images_all.reshape((images_all.shape[0], 128*128))

# if SUFFIX != "hoge":
#     n, h, w = images_tr.shape
#     images = images_tr.reshape(n, h*w)

print(images_tr.shape)



images_std, mu, sigma = standardize(images_all, full=True)
# print(images_std.shape)
pca = PCA(n_components=256)
pca.fit(images_std)
n, perc = min(enumerate(pca.explained_variance_ratio_.cumsum()), key=lambda x: abs(x[1]-PERC_VARIANCE))
print(f"Number of principal components explaining {100*PERC_VARIANCE}% of variance: {n}, {perc}")

fig, ax = plt.subplots(1,1,figsize=(15,3), dpi=200)
sns.lineplot(x=range(1, 257), y=pca.explained_variance_ratio_.cumsum(), ax=ax);
fig.suptitle('Total Variation Explained by Number of PCA Components', fontsize='xx-large')
ax.set_xlabel('Number of Components', fontsize='x-large')
ax.set_ylabel('% Variation Explained', fontsize='x-large')
plt.axvline(min(enumerate(pca.explained_variance_ratio_.cumsum()), key=lambda x: abs(x[1]-PERC_VARIANCE))[0], dashes=(1.0, 1.0))
plt.axhline(min(enumerate(pca.explained_variance_ratio_.cumsum()), key=lambda x: abs(x[1]-PERC_VARIANCE))[1], dashes=(1.0, 1.0))
plt.savefig(f"{SUFFIX}_pca_var_plot.png")



images_tr = standardize(images_tr, mu, sigma)
pcas = pca.transform(images_tr)
pcas[:, -(280-n):] = 0
save_zarr(city=CITY, data=pcas, suffix=f"pca_{SUFFIX}_tr", path=DATA_DIR)

images_va = standardize(images_va, mu, sigma)
pcas = pca.transform(images_va)
pcas[:, -(280-n):] = 0
save_zarr(city=CITY, data=pcas, suffix=f"pca_{SUFFIX}_va", path=DATA_DIR)

images_te = standardize(images_te, mu, sigma)
pcas = pca.transform(images_te)
pcas[:, -(280-n):] = 0
save_zarr(city=CITY, data=pcas, suffix=f"pca_{SUFFIX}_te", path=DATA_DIR)

fig,ax = plt.subplots(2,5,sharex=True,sharey=True,figsize=(8,3), dpi=300)
axes = ax.flatten()
pcas = pca.transform(images_va[:5])
pcas[:, -(280-n):] = 0
for i, im in enumerate(pcas):
    result = pca.inverse_transform(im).reshape((128,128))
    axes[i].imshow(images_va[i].reshape(128,128), cmap="gray")
    axes[i+5].imshow(result, cmap="gray")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(f"Images reconstructed using {n} principal components; {PERC_VARIANCE*100}% variance retained",fontsize=12)
plt.tight_layout()
plt.savefig(f"{SUFFIX}_reconstructed.png")
