import os
import numpy as np
from skimage.feature import hog
import rasterio



CITY = 'aleppo_cropped'
image_dir = f"../../../data/{CITY}/images"
# image_dir = f"/Users/arogyak/projects/mwd/v2_des/experiments/cropping"
images = sorted([f for f in os.listdir(image_dir) if ".tif" in f])
# images = images[11:]
dest = os.getcwd()
print(dest)

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

def get_hog(image):
    image = np.float32(image)
    _, image = hog(image, orientations=9, pixels_per_cell=(2,2),
                	cells_per_block=(4, 4), channel_axis=2, visualize=True)
    return _, image     

for image in images:
    destination = f'{dest}/{image}'
    source = f"{image_dir}/{image}"
    source, profile = read_raster(source, profile=True)
    source = get_hog(source)
    write_raster(source, destination=destination, profile=profile)
    print(_.shape)
