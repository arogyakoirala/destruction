import os
import re
import numpy as np
import geopandas
import rasterio
from rasterio import features

CITY = 'aleppo_cropped'
TILE_SIZE = [128,128]
REFRESH_SAMPLE = True
DATA_DIR = "../../../data"

def search_data(pattern:str='.*', directory:str='../data') -> list:
    '''Sorted list of files in a directory matching a regular expression'''
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    if len(files) == 1: files = files[0]
    return files

def pattern(city:str='.*', type:str='.*', date:str='.*', ext:str='tif') -> str:
    '''Regular expressions for search_data'''
    return f'^.*{city}/.*/{type}_{date}\.{ext}$'

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

# Locate files
image      = search_data(pattern(city=CITY, type='image'), directory=DATA_DIR)[0]
settlement = search_data(f'{CITY}_settlement.*gpkg$', directory=DATA_DIR)
noanalysis = search_data(f'{CITY}_noanalysis.*gpkg$', directory=DATA_DIR)

profile    = tiled_profile(image, tile_size=(*TILE_SIZE, 1))
settlement = rasterise(settlement, profile, dtype='bool')
noanalysis = rasterise(noanalysis, profile, dtype='bool')
analysis   = np.logical_and(settlement, np.invert(noanalysis))


del image, settlement, noanalysis

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


# Generate samples if REFRESH_SAMPLES=True
if REFRESH_SAMPLE:
    # Splits samples
    np.random.seed(42)
    index   = dict(training=0.70, validation=0.15, test=0.15)
    index   = np.random.choice(np.arange(len(index)) + 1, np.sum(analysis), p=list(index.values()))
    samples = analysis.astype(int)
    np.place(samples, analysis, index)
    write_raster(samples, profile, f'{DATA_DIR}/{CITY}/others/{CITY}_samples.tif', nodata=-1, dtype='int8')
    del index, samples, analysis
