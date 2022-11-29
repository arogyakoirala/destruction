import os
import rasterio
from rasterio import features, windows
import numpy as np

TILE_SIZE = (128, 128)
WINDOW_SIZE = (5,5)
# XOFFSET = 85
# YOFFSET = -45
# XOFFSET = 60 5by5
# YOFFSET = -60
XOFFSET = 0
YOFFSET = 0
CITY = 'aleppo_cropped'

image_dir = f"../../../data/{CITY}/images"
# image_dir = os.getcwd()
images = sorted([f for f in os.listdir(image_dir) if ".tif" in f])
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

def center_window(source:str, size:dict, tile_size:dict=(128,128), xoffset:int=0, yoffset:int=0):
    '''Computes the windows for the centre of a raster'''
    profile = rasterio.open(source).profile
    xoffset = xoffset * tile_size[0]
    yoffset = yoffset * tile_size[1]
    centre  = ((profile['width'] // 2), (profile['height'] // 2))
    window  = windows.Window.from_slices(
        (centre[0] - (size[0] // 2) + yoffset, centre[0] + size[0] // 2 + yoffset),
        (centre[1] - size[1] // 2 + xoffset, centre[1] + size[1] // 2 + xoffset)
    )
    return window

for image in images:
    source = f"{image_dir}/{image}"
    window = center_window(source, 
            (WINDOW_SIZE[0]*TILE_SIZE[0], WINDOW_SIZE[1]*TILE_SIZE[1]), TILE_SIZE,
                xoffset=XOFFSET, yoffset=YOFFSET)
    with rasterio.open(source) as src:
        # window = window
        # window = Window(col_off=13, row_off=3, width=757, height=711)

        kwargs = src.meta.copy()
        kwargs.update({
            'height': window.height,
            'width': window.width,
            'transform': rasterio.windows.transform(window, src.transform)})
        destination = f'{dest}/{image}'
        if os.path.exists(destination):
            os.remove(destination)
        with rasterio.open(destination, 'w', **kwargs) as dst:
            dst.write(src.read(window=window))
    print(image)