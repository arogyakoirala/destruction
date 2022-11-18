import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--city", help="city")
parser.add_argument("--pre_image_index", help="index of images to use as pre image")
parser.add_argument('--refresh_sample', default=False, action="store_true",
                    help="Regenerate sample?")
parser.add_argument('--data_dir', help="Path to data folder")
args = parser.parse_args()

import os
import sys
from matplotlib import pyplot
import time
import geopandas
import pandas as pd
import numpy as np

from scripts import destruction_utilities as utils

DEBUG = False
CITY = 'aleppo'
TILE_SIZE = [128,128]
REFRESH_SAMPLE = True
ZERO_DAMAGE_BEFORE_YEAR = 2012
PRE_IMAGE_INDEX = [0]

DATA_DIR = "../data"


if args.city:
    CITY = args.city

if args.pre_image_index:
    PRE_IMAGE_INDEX = [int(el.strip()) for el in args.pre_image_index.split(",")]

if args.refresh_sample:
    REFRESH_SAMPLE = args.refresh_sample

if args.data_dir:
    DATA_DIR = args.data_dir
    
print(f"Run parameters: city={CITY}, refresh_sample={REFRESH_SAMPLE}, pre_image_index={PRE_IMAGE_INDEX}, data_dir={DATA_DIR} ...")


# Locate files
image      = utils.search_data(utils.pattern(city=CITY, type='image'), directory=DATA_DIR)[0]
settlement = utils.search_data(f'{CITY}_settlement.*gpkg$', directory=DATA_DIR)
noanalysis = utils.search_data(f'{CITY}_noanalysis.*gpkg$', directory=DATA_DIR)


profile    = utils.tiled_profile(image, tile_size=(*TILE_SIZE, 1))
settlement = utils.rasterise(settlement, profile, dtype='bool')
noanalysis = utils.rasterise(noanalysis, profile, dtype='bool')
analysis   = np.logical_and(settlement, np.invert(noanalysis))

del image, settlement, noanalysis

# Generate samples if REFRESH_SAMPLES=True
if REFRESH_SAMPLE:
    # Splits samples
    np.random.seed(1)
    index   = dict(training=0.70, validation=0.15, test=0.15)
    index   = np.random.choice(np.arange(len(index)) + 1, np.sum(analysis), p=list(index.values()))
    samples = analysis.astype(int)
    np.place(samples, analysis, index)
    utils.write_raster(samples, profile, f'{DATA_DIR}/{CITY}/others/{CITY}_samples.tif', nodata=-1, dtype='int8')
    del index, samples, analysis
    
# Reads damage reports
damage = utils.search_data(f'{CITY}_damage.*gpkg$', directory=DATA_DIR)
damage = geopandas.read_file(damage)
last_annotation_date = sorted(damage.columns)[-2]

# Extract report dates
dates = utils.search_data(utils.pattern(city=CITY, type='image'), directory=DATA_DIR)
dates = utils.extract(dates, '\d{4}_\d{2}_\d{2}')
dates= list(map(lambda x: x.replace("_", "-"), dates))

# add additional date columns
known_dates = sorted(damage.drop('geometry', axis =1).columns)
damage[list(set(dates) - set(damage.columns))] = np.nan
damage = damage.reindex(sorted(damage.columns), axis=1)

# Set pre cols to 0
pre_cols = [col for col in sorted(damage.drop('geometry', axis=1).columns) if int(col.split("-")[0]) < ZERO_DAMAGE_BEFORE_YEAR]

for i, col in enumerate(sorted(damage.drop('geometry', axis=1).columns)):
    if i in PRE_IMAGE_INDEX and col not in pre_cols:
        pre_cols.append(col)
        
damage[pre_cols] = 0.0

# Get post cols
post_cols = sorted([col for col in damage.drop('geometry', axis=1).columns if col not in pre_cols])

# Fill uncertains between two dates
last_known_date = known_dates[0]
for col in post_cols:
    if col in known_dates and time.strptime(col, "%Y-%m-%d") >= time.strptime(last_known_date, "%Y-%m-%d"):
        last_known_date = col
        if(known_dates.index(col) < len(known_dates)-1):
            next_known_date = known_dates[known_dates.index(col)+1]
            dates_between = post_cols[post_cols.index(last_known_date)+1:post_cols.index(next_known_date)]
            zeros = list(*np.where(damage[next_known_date] == 0.0))
            not_equal = list(*np.where(damage[last_known_date] != damage[next_known_date]))
            for date in dates_between:
                damage.loc[not_equal, date] = -1
                
# Backfill the zeros
filled = []
last_known_date = None
for j, col in enumerate(post_cols):
    zeros = list(*np.where(damage[col] == 0.0))
    cols_before_date = [c for c in post_cols if time.strptime(c, "%Y-%m-%d")  < time.strptime(col, "%Y-%m-%d") ]
    for i, date in enumerate(cols_before_date):       
        if date not in filled and date not in known_dates:
            zeros = list(*np.where(damage[col] == 0.0))
            uncertains = list(*np.where(damage[date] != -1))
            n_uncertains = list(set(zeros).intersection(set(uncertains)))
            damage.loc[n_uncertains, date] = 0.0
            filled.append(date) 
            
# Label the uncertain class everywhere now
geometry = damage.geometry
damage_ = damage.drop('geometry', axis=1)
damage_ = damage_.T
for col in damage_.columns:
    uncertains = np.where(damage_[col].fillna(method='ffill') != damage_[col].fillna(method='bfill'))
    damage_.iloc[uncertains, col] = -1
damage = damage_.T
damage['geometry'] = geometry
damage = geopandas.GeoDataFrame(damage)

# Forward fill the rest
geometry = damage.geometry
damage_ = damage.drop('geometry', axis=1)
damage_ = damage_.T
damage_ = damage_.fillna(method='ffill')
damage = damage_.T
damage['geometry'] = geometry
damage = geopandas.GeoDataFrame(damage)

# Writes damage labels
for date in damage.drop('geometry', axis=1).columns:
    print(f'------ {date}')
    subset = damage[[date, 'geometry']].sort_values(by=date) # Sorting takes the max per pixel
    subset = utils.rasterise(subset, profile, date)
    utils.write_raster(subset, profile, f'{DATA_DIR}/{CITY}/labels/label_{date}.tif', nodata=-1, dtype='int8')
del date, subset
