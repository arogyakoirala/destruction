import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--suffix", help="Suffix")
args = parser.parse_args()

SUFFIX = 'lap'
CITY = 'aleppo_cropped'
DATA_DIR = "../../../data"

if args.suffix:
    SUFFIX = args.suffix 


from matplotlib import cm
import numpy as np
from sklearn.manifold import TSNE
import zarr
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox



def read_zarr(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    return zarr.open(path)

def getImage(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=zoom)

images_tr = read_zarr(CITY, f"{SUFFIX}_tr_post", DATA_DIR)[:]
images_va = read_zarr(CITY, f"{SUFFIX}_va_post", DATA_DIR)[:]
images_te = read_zarr(CITY, f"{SUFFIX}_te_post", DATA_DIR)[:]

print(images_tr.shape)

images_tr = images_tr.reshape((images_tr.shape[0], 128*128))
images_va = images_va.reshape((images_va.shape[0], 128*128))
images_te = images_te.reshape((images_te.shape[0], 128*128))

images_all = np.append(images_tr, images_va, axis=0)
images_all = np.append(images_all, images_te, axis=0)


# # TSNE (dimensionality reduction)

tsne = TSNE(n_components=2, random_state=0)
tsne    = tsne.fit_transform(images_all) # 2D representation
X = tsne[:, 0]
Y = tsne[:, 1]
# y    = range(len(digits.target_names)) # labels for visual




images_tr = read_zarr(CITY, f"{SUFFIX}_tr_post", DATA_DIR)[:]
images_va = read_zarr(CITY, f"{SUFFIX}_va_post", DATA_DIR)[:]
images_te = read_zarr(CITY, f"{SUFFIX}_te_post", DATA_DIR)[:]

# images_tr = images_tr.reshape((images_tr.shape[0], 128*128))
# images_va = images_va.reshape((images_va.shape[0], 128*128))
# images_te = images_te.reshape((images_te.shape[0], 128*128))

images_all = np.append(images_tr, images_va, axis=0)
images_all = np.append(images_all, images_te, axis=0)

# fig,ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=(8,8), dpi=300)
fig, ax = plt.subplots(1,1,figsize=(30,30),dpi=100)
ax.scatter(X, Y) 
i = 0
for x0, y0 in zip(X, Y):
    # ab = AnnotationBbox(getImage(images_all[0].reshape((128,128))), (x0, y0), frameon=False)
    arr = np.arange(100).reshape((10, 10))
    arr = images_all[i]
    i +=1
    im = OffsetImage(arr, zoom=0.2, cmap="gray")
    im.image.axes = ax
    ab = AnnotationBbox(im, [x0, y0],
                        # xybox=(-50., 50.),
                        # xycoords='data',
                        # boxcoords="offset points",
                        # pad=0.3,
                        # arrowprops=dict(arrowstyle="->")
                        )
    ax.add_artist(ab)
ax.set_facecolor("yellow")

plt.savefig(f"tsne_{SUFFIX}.png")