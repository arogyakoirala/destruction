import numpy as np
import rasterio
from rasterio import features
import os
import re
import geopandas
import time


CITY = "raqqa"
DATA_DIR = "../data"
ZERO_DAMAGE_BEFORE_YEAR = 2012
# PRE_IMAGE_INDEX = [0,1]
TILE_SIZE = (128,128)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--city", help="City")
parser.add_argument("--data_dir", help="Data Dir")
args = parser.parse_args()

if args.city:
    CITY = args.city

if args.data_dir:
    DATA_DIR = args.data_dir

def search_data(pattern:str='.*', directory:str='../data') -> list:
    '''Sorted list of files in a directory matching a regular expression'''
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    # if len(files) == 1: files = files[0]
    return files

def pattern(city:str='.*', type:str='.*', date:str='.*', ext:str='tif') -> str:
    '''Regular expressions for search_data'''
    return f'^.*{city}/.*/{type}_{date}\.{ext}$'

def read_raster(source:str, band:int=None, window=None, dtype:str='int', profile:bool=False) -> np.ndarray:
    '''Reads a raster as a numpy array'''
    raster = rasterio.open(source)
    if band is not None:
        image = raster.read(band, window=window)
        image = np.expand_dims(image, 0)
    else: 
        image = raster.read(window=window)
    image = image.transpose([1, 2, 0]).astype(dtype)
    if profile:
        return image, raster.profile
    else:
        return image

def extract(files:list, pattern:str='\d{4}-\d{2}-\d{2}') -> list:
    pattern = re.compile(pattern)
    match   = [pattern.search(file).group() for file in files]
    return match

def rasterise(source, profile, attribute:str=None, dtype:str='float32') -> np.ndarray:
    '''Tranforms vector data into raster'''
    if isinstance(source, str): 
        source = geopandas.read_file(source)
    if isinstance(profile, str): 
        profile = rasterio.open(profile).profile
    geometries = source['geometry']
    if attribute is not None:
        geometries = zip(geometries, source[attribute])
    image = features.rasterize(geometries, out_shape=(profile['height'], profile['width']), transform=profile['transform'])
    image = image.astype(dtype)
    return image

def write_raster(array:np.ndarray, profile, destination:str, nodata:int=None, dtype:str='float64') -> None:
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

def tiled_profile(source:str, tile_size:tuple=(128,128,1)) -> dict:
    '''Computes raster profile for tiles'''
    raster  = rasterio.open(source)
    profile = raster.profile
    assert profile['width']  % tile_size[0] == 0, 'Invalid dimensions'
    assert profile['height'] % tile_size[1] == 0, 'Invalid dimensions'
    affine  = profile['transform']
    affine  = rasterio.Affine(affine[0] * tile_size[0], affine[1], affine[2], affine[3], affine[4] * tile_size[1], affine[5])
    profile.update(width=profile['width'] // tile_size[0], height=profile['height'] // tile_size[0], count=tile_size[2], transform=affine)
    return profile

image      = search_data(pattern(city=CITY, type='image'), directory=DATA_DIR)[0]
profile    = tiled_profile(image, tile_size=(*TILE_SIZE, 1))

# Reads damage reports
damage = search_data(f'{CITY}_damage.*gpkg$', directory=DATA_DIR)[0]
damage = geopandas.read_file(damage)
last_annotation_date = sorted(damage.columns)[-2]

# Extract report dates
dates = search_data(pattern(city=CITY, type='image'), directory=DATA_DIR)
dates = extract(dates, '\d{4}_\d{2}_\d{2}')
dates = list(map(lambda x: x.replace("_", "-"), dates))

# add additional date columns
known_dates = sorted(damage.drop('geometry', axis =1).columns)
damage[list(set(dates) - set(damage.columns))] = np.nan
damage = damage.reindex(sorted(damage.columns), axis=1)

# Set pre cols to 0
# pre_cols = [col for col in sorted(damage.drop('geometry', axis=1).columns) if int(col.split("-")[0]) < ZERO_DAMAGE_BEFORE_YEAR]
pre_cols = []
pre_images  = search_data(pattern='^.*tif', directory=f'{DATA_DIR}/{CITY}/images/pre')
pre_cols = [f.split("image_")[1].split(".tif")[0].replace("_", "-") for f in pre_images]

# for i, col in enumerate(sorted(damage.drop('geometry', axis=1).columns)):
#     if col not in pre_cols:
#         pre_cols.append(col)

damage[pre_cols] = 0.0

f = open(f"{DATA_DIR}/{CITY}/others/metadata.txt", "a")

f.write("\n\n######## Labeling Step\n\n")
f.write(f"Using {pre_cols} as pre-dates\n")

# Get post cols
post_cols = sorted([col for col in damage.drop('geometry', axis=1).columns if col not in pre_cols])
f.write(f"Considering {post_cols} as post-dates")


# Corrected label assignment
geom = damage['geometry']
damage = damage.drop('geometry', axis=1).T
for col in damage.columns:
    unc = np.where(damage[col].fillna(method='ffill') != damage[col].fillna(method='bfill'))
    if int(col)%1000 == 0:
        print(col)

    for i in unc:
        damage[col][i] = -1

for col in damage.columns:
    damage[col] = damage[col].fillna(method='ffill')

damage = damage.T
damage['geometry'] = geom
damage = damage

# Writes damage labels
for date in damage.drop('geometry', axis=1).columns:
    print(f'------ {date}')
    subset = damage[[date, 'geometry']].sort_values(by=date) # Sorting takes the max per pixel
    subset[date] = np.where((subset[date] == -1.0) , 99.0, subset[date])
    subset[date] = np.where((subset[date] < 3), 0.0, subset[date])
    subset[date] = np.where((subset[date] == 3), 1.0, subset[date])
    subset = rasterise(subset, profile, date)
    write_raster(subset, profile, f'{DATA_DIR}/{CITY}/labels/label_{date}.tif', dtype='int8')
del date, subset

f.close()