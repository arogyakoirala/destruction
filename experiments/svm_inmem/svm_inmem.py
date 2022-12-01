import os 
import rasterio
import numpy as np 
from sklearn import svm
import re
import time 
import random
import zarr
import shutil


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix




CITY = 'aleppo_cropped'
DATA_DIR = "../../../data"
TILE_SIZE = (128,128)
SUFFIX = 'pca_lap'

def read_zarr(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    print(path)
    return zarr.open(path)

def standardize(data, mu=None, sigma=None, full = False):
    if mu is None:
        mu = np.mean(data, axis=0)
    if sigma is None:
        sigma = np.std(data, axis=0) + 0.00000000000001
    if not full:
        return ((data - mu) / sigma)
    return ((data - mu) / sigma), mu, sigma

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


images_tr_post = read_zarr(CITY, f"{SUFFIX}_tr_post", DATA_DIR)[:]
images_va_post = read_zarr(CITY, f"{SUFFIX}_va_post", DATA_DIR)[:]
images_te_post = read_zarr(CITY, f"{SUFFIX}_te_post", DATA_DIR)[:]
labels_tr_post = read_zarr(CITY, f"la_tr", DATA_DIR)[:]
labels_va_post = read_zarr(CITY, f"la_va", DATA_DIR)[:]
labels_te_post = read_zarr(CITY, f"la_te", DATA_DIR)[:]


images_tr_post = np.append(images_tr_post, images_va_post, axis=0)
labels_tr_post = np.append(labels_tr_post, labels_va_post, axis=0)

# # create model
# model = svm.SVC(gamma="auto", verbose=True)
# cv = KFold(n_splits=10, random_state=1, shuffle=True)
# scores = cross_val_score(model, images_tr_post, labels_tr_post, scoring="accuracy", cv=cv, n_jobs=-1)
# # report performance
# print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# # yhat = model.predict(images_te_post)
# # print(yhat)

# print(scores)


# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100], 
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid.fit(images_tr_post, labels_tr_post)


# print best parameter after tuning
print(grid.best_params_)
  
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

grid_predictions = grid.predict(images_te_post)
  
# print classification report
print(classification_report(labels_te_post, grid_predictions))