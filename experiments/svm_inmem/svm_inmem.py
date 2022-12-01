import os 
import rasterio
import numpy as np 
from sklearn import svm
import re
import time 
import random
import zarr
import shutil
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score

import matplotlib.pyplot as plt
from joblib import dump, load
from datetime import datetime






CITY = 'aleppo_cropped'
DATA_DIR = "../../../data"
MODEL_DIR = "../../../data"
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

print(images_tr_post.shape)
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
param_grid = {'C': [0.3, 1, 3], 
              'gamma': [10,3, 1,0.3, 0.1],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(svm.SVC(probability=True), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid.fit(images_tr_post, labels_tr_post)


# print best parameter after tuning
print(grid.best_params_)
  
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

MODEL_CODE = f"snn_{SUFFIX}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
MODEL_STORAGE_LOCATION = f"{MODEL_DIR}/{MODEL_CODE}"
Path(MODEL_STORAGE_LOCATION).mkdir(parents=True, exist_ok=True)

estimator = grid.best_estimator_
dump(estimator, f"{MODEL_STORAGE_LOCATION}/model.joblib")
# Somewhere else
# estimator = load("your-model.joblib")

yhat = grid.predict(images_te_post)
yhat_proba = grid.predict_proba(images_te_post)[:, 1]




  
# print classification report
print(classification_report(labels_te_post, yhat))


roc_auc_test = roc_auc_score(labels_te_post, yhat_proba)
#calculate precision and recall
precision, recall, thresholds = precision_recall_curve(labels_te_post, yhat_proba)
# F1 = 2 * (precision * recall) / (precision + recall)
# p_score = precision_score(y, yhat_proba)
# r_score = recall_score(y, yhat_proba)
# plot_roc_curve(fpr, tpr)
print(f'Run {MODEL_STORAGE_LOCATION}: \nTest Set AUC Score for the ROC Curve: {roc_auc_test} \nAverage precision:  {np.mean(precision)}' )



#create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')

#add axis labels to plot
ax.set_title('Precision - Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')


#display plot
plt.savefig(f"{MODEL_DIR}/pr_curve_{SUFFIX}.png")