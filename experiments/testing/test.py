import geopandas as gpd
import zarr
import numpy as np
copy = "aleppo_damage_col_drop.gpkg"
orig = "aleppo_damage.gpkg"


# copy = gpd.read_file(copy)
# orig = gpd.read_file(orig)


# print(orig.describe())
# print(copy.describe())


# print(len(orig), len(copy))
# # print(copy.describe())


# CITY = 'aleppo_cropped'
# SUFFIX = 'hog'
# BANDS = 1
# N_BANDS = 3
# DATA_DIR = "../../../data"
# MODEL_DIR = "../../../models"
# BATCH_SIZE = 32
# PATCH_SIZE = (128,128)
# FILTERS = [32]
# DROPOUT = [0.3, 0.45]
# EPOCHS = [70, 100]
# UNITS = [64]
# LR = [0.003, 0.003, 0.004]


def read_zarr(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    return zarr.open(path)


# print(read_zarr(CITY, "la_tr", DATA_DIR)[:])

la_tr = read_zarr(CITY, "unb_la_tr", DATA_DIR )[:]
la_va = read_zarr(CITY, "unb_la_va", DATA_DIR )[:]
la_te = read_zarr(CITY, "unb_la_te", DATA_DIR )[:]

print(np.unique(la_tr, return_counts=True))
print(np.unique(la_va, return_counts=True))
print(np.unique(la_te, return_counts=True))
