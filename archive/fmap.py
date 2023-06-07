from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import AUC
from tensorflow.keras import backend as K
import numpy as np
import math
import rasterio
from pathlib import Path
import os
import re





OUTPUT_DIR = "../outputs"
# CITIES = ['aleppo', 'damascus']
DATA_DIR = "../data/destr_data"
TILE_SIZE = (128,128)



import argparse
parser = argparse.ArgumentParser()
# parser.add_argument("--cities", help="Cities, comma separated. Eg: aleppo,raqqa,damascus")
parser.add_argument("run_id", help="Model Run ID for which we want to generate predictions")
args = parser.parse_args()

# if args.cities:
#     CITIES = [el.strip() for el in args.cities.split(",")]


RUN_ID = int(args.run_id)
RUN_DIR = f'{OUTPUT_DIR}/{RUN_ID}'
MODEL_PATH = f'{RUN_DIR}/model'
PRED_DIR = f'{OUTPUT_DIR}/{RUN_ID}/predictions'

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


auc = AUC(
    num_thresholds=200,
    curve='ROC',
    name = 'auc'
)


model = load_model(MODEL_PATH, custom_objects={'f1_m':f1_m, 'precision_m': precision_m, 'recall_m': recall_m, 'auc': auc, 'K': K})


# summarize feature map shapes
for i in range(len(model.layers)):
    layer = model.layers[i]
    # check for convolutional layer
    if 'conv' not in layer.name:
        continue
    # summarize output shape
    print(i, layer.name, layer.output.shape)
