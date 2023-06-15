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
import pandas as pd
import gc
from sys import getsizeof
## For artemisa
# CITIES = ['aleppo', 'damascus', 'daraa', 'deir-ez-zor','hama', 'homs', 'idlib', 'raqqa']
# OUTPUT_DIR = "/lustre/ific.uv.es/ml/iae091/outputs"
# DATA_DIR = "/lustre/ific.uv.es/ml/iae091/data"

## For workstation
CITIES = ['aleppo', 'damascus', 'daraa', 'deir-ez-zor','hama', 'homs', 'idlib', 'raqqa']
OUTPUT_DIR = "../outputs"
DATA_DIR = "../data"

## For local
# CITIES = ['aleppo', 'daraa']
# OUTPUT_DIR = "../data/destr_outputs"
# DATA_DIR = "../data/destr_data"

TILE_SIZE = (128,128)



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cities", help="Cities, comma separated. Eg: aleppo,raqqa,damascus")
parser.add_argument("run_id", help="Model Run ID for which we want to generate predictions")

args = parser.parse_args()

if args.cities:
    CITIES = [el.strip() for el in args.cities.split(",")]


RUN_ID = int(args.run_id)
RUN_DIR = f'{OUTPUT_DIR}/{RUN_ID}'
MODEL_PATH = f'{RUN_DIR}/model'
PRED_DIR = f'{OUTPUT_DIR}/{RUN_ID}/predictions'
SAVE_RASTER = False



def make_tuple_pair(n, step_size):
    if step_size > n:
        return [(0,n)]
    iters = math.ceil(n/step_size*1.0)
    l = []
    for i in range(0, iters):
        if i == iters - 1:
            t = (i*step_size, n)
            l.append(t)
        else:
            t = (i*step_size, (i+1)*step_size)
            l.append(t)
    return l

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

def search_data(pattern:str='.*', directory:str='../data') -> list:
    '''Sorted list of files in a directory matching a regular expression'''
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    if len(files) == 1: files = files
    return files


class SiameseGenerator(Sequence):
    def __init__(self, images, labels=None, batch_size=1, train=True):
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

        if self.train:
            y = self.labels[index*self.batch_size:(index+1)*self.batch_size]
            return {'images_t0': X_pre, 'images_tt': X_post}, y
        else:
            return {'images_t0': X_pre, 'images_tt': X_post}

def read_raster(source:str, band:int=None, window=None, dtype:str='uint8', profile:bool=False) -> np.ndarray:
    '''Reads a raster as a numpy array'''
    raster = rasterio.open(source)
    if band is not None:
        image = raster.read(band, window=window)
        image = np.expand_dims(image, 0)
    else: 
        image = raster.read(window=window)
    # print(image.shape)
    # image = image.transpose([1, 2, 0]).astype(dtype)
    image = image.transpose([1, 2, 0]).astype(dtype)
    if profile:
        return image, raster.profile
    else:
        return image


def write_raster(array:np.ndarray, profile, destination:str, nodata:int=None, dtype:str='float32') -> None:
    '''Writes a numpy array as a raster'''
    if array.ndim == 2:
        array = np.expand_dims(array, 2)
    array = array.transpose([2, 0, 1]).astype(dtype)
    bands, height, width = array.shape
    if isinstance(profile, str):
        profile = rasterio.open(profile).profile
    profile.update(driver='GTiff', dtype=dtype, count=bands, nodata=nodata)
    with rasterio.open(fp=destination, mode='w', **profile) as raster:
        raster.write(array)
        raster.close()

def tile_sequences(images:np.ndarray, tile_size:tuple=(128, 128)) -> np.ndarray:
    '''Converts images to sequences of tiles'''
    n_images, image_height, image_width, n_bands = images.shape
    tile_width, tile_height = tile_size
    assert image_width  % tile_width  == 0
    assert image_height % tile_height == 0
    n_tiles_width  = (image_width  // tile_width)
    n_tiles_height = (image_height // tile_height)
    sequence = images.reshape(n_images, n_tiles_width, tile_width, n_tiles_height, tile_height, n_bands)
    sequence = np.moveaxis(sequence.swapaxes(2, 3), 0, 2)
    sequence = sequence.reshape(-1, n_images, tile_width, tile_height, n_bands)
    return sequence


def tiled_profile(source:str, tile_size:tuple=(*TILE_SIZE, 1)) -> dict:
    '''Computes raster profile for tiles'''
    raster  = rasterio.open(source)
    profile = raster.profile
    assert profile['width']  % tile_size[0] == 0, 'Invalid dimensions'
    assert profile['height'] % tile_size[1] == 0, 'Invalid dimensions'
    affine  = profile['transform']
    affine  = rasterio.Affine(affine[0] * tile_size[0], affine[1], affine[2], affine[3], affine[4] * tile_size[1], affine[5])
    profile.update(width=profile['width'] // tile_size[0], height=profile['height'] // tile_size[0], count=tile_size[2], transform=affine)
    return profile

auc = AUC(
    num_thresholds=200,
    curve='ROC',
    name = 'auc'
)

best_model = load_model(MODEL_PATH, custom_objects={'f1_m':f1_m, 'precision_m': precision_m, 'recall_m': recall_m, 'auc': auc, 'K': K})


final_df = None


for city in CITIES:
    print(city)
    pre_images  = search_data(pattern='^.*tif', directory=f'{DATA_DIR}/{city}/images/pre')
    post_images  = search_data(pattern='^.*tif', directory=f'{DATA_DIR}/{city}/images/post')

    # print(pre_images)
    # print(post_images)

    _pred_dir = f"{PRED_DIR}/{city}"
    Path(_pred_dir).mkdir(parents=True, exist_ok=True)


    for j, pre in enumerate(pre_images):
        
        date_pre = pre.split("/")[-1].split("image_")[1].split(".tif")[0].replace("_", "-")
        
        


        for i in range(len(post_images)):
            image = post_images[i]

            date_post = image.split("/")[-1].split("image_")[1].split(".tif")[0].replace("_", "-")
            label_path = f"{DATA_DIR}/{city}/labels/label_{date_post}.tif"

            pre_image = read_raster(pre)
            print(f"{city} - Using pre image: {date_pre}")

            pre_image = tile_sequences(np.array([pre_image]), TILE_SIZE)
            pre_image = np.squeeze(pre_image) / 255.0
            
            profile = tiled_profile(image, tile_size=(*TILE_SIZE, 3))
            image = read_raster(image)

            image = tile_sequences(np.array([image]))
            image = np.squeeze(image) / 255.0
            x = SiameseGenerator((pre_image, image), train=False)
            yhat = best_model.predict(x)
            y = read_raster(label_path)

            temp_df = pd.DataFrame()
            temp_df['y'] = y.flatten().tolist()
            temp_df['yhat'] = yhat.flatten().tolist()
            temp_df['pre'] = date_pre
            temp_df['post'] = date_post
            temp_df['city'] = city

            # if final_df is None:
            #     final_df = temp_df
            # else:
            #     final_df = pd.concat([final_df, temp_df], ignore_index=True)



            # print(labels)
            if SAVE_RASTER:
                print("Saving raster...")
                write_raster(yhat.reshape((profile['height'], profile['width'])), profile, f"{_pred_dir}/pred_{post_images[i].split('image_')[1]}") 
            print(f"\t - {date_post} predictions completed")
            print(f"\t\t - yhat samples: {yhat.flatten().tolist()[0:5]}")


            out_csv_path = f"{RUN_DIR}/actual_v_predicted_{RUN_ID}_{city}.csv"
            if os.path.exists(out_csv_path):
                temp_df.to_csv(out_csv_path, mode="a", index=False)
            else:
                temp_df.to_csv(out_csv_path, index=False)

            print("\t\t - Garbage collection")
            list_of_locals = list(locals())
            for var in list_of_locals:
                if var == "pre_image":
                    print("\t\t - variable {} has size {}".format(var,getsizeof(locals()[var])))
                if var == "image":
                    print("\t\t - variable {} has size {}".format(var,getsizeof(locals()[var])))
            del pre_image, image, profile, x, y, temp_df

           
            gc.collect()
            print("\t\t - Garbage collection complete")

    final_df = None