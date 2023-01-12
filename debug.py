import zarr
from tensorflow.keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt


CITY = 'aleppo'
DATA_DIR = "../data"

class SiameseGenerator(Sequence):
    def __init__(self, images, labels, batch_size=32, train=True):
        self.images_pre = images[0]
        self.images_post = images[1]
        self.labels = labels
        self.batch_size = batch_size
        self.train = train


        
        # self.tuple_pairs = make_tuple_pair(self.images_t0.shape[0], int(self.batch_size/4))
        # np.random.shuffle(self.tuple_pairs)
    def __len__(self):
        return len(self.images_pre)//self.batch_size    
    
    def __getitem__(self, index):
        X_pre = self.images_pre[index*self.batch_size:(index+1)*self.batch_size]
        X_post = self.images_post[index*self.batch_size:(index+1)*self.batch_size]
        y = self.labels[index*self.batch_size:(index+1)*self.batch_size]

        if self.train:
            return {'images_t0': X_pre, 'images_tt': X_post}, y
        else:
            return {'images_t0': X_pre, 'images_tt': X_post}

def read_zarr(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    return zarr.open(path)


pre = read_zarr(CITY, "im_tr_pre")
post = read_zarr(CITY, "im_tr_post")
y = read_zarr(CITY, "la_tr")


gen = SiameseGenerator((pre, post), y)

indices = list(np.where(y[:]==1)[0])
print(f"{len(indices)} positive samples..")
indices = np.random.choice(indices, 5)//32

for j, ind in enumerate(indices):
    fig, ax = plt.subplots(2,8,dpi=400, figsize=(25,6))
    ax = ax.flatten()
    for i, image in enumerate(gen.__getitem__(ind)[0]['images_t0'][0:8]):
        ax[i].imshow(image)
        ax[i].set_title(gen.__getitem__(ind)[1][i] == 1)
    for i, image in enumerate(gen.__getitem__(ind)[0]['images_tt'][0:8]):
        ax[i+8].imshow(image)
    plt.suptitle("Training set (sample images; top=pre, bottom=post)")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/{CITY}/others/debug_samples_{j+1}.png")



