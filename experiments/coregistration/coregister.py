import os
from arosics import COREG
import rasterio

CITY = 'aleppo_cropped'

images_dir = f"../../../data/{CITY}/images"

images = sorted([f for f in os.listdir(images_dir) if ".tif" in f])
im_reference = f'{images_dir}/{images[1]}'

failures = []
def correct(image):
    im_target    = f'{images_dir}/{image}'
    destination  = f'{os.getcwd()}/{image}'.split(".tif")[0]+".bsq"
    if os.path.exists(destination):
        os.remove(destination)
    # CR = COREG(im_reference, im_target, path_out=destination, wp=(37.179159,36.194390), max_iter=5000, fmt_out='ENVI', max_shift=200)
    CR = COREG(im_reference, im_target, path_out=destination, wp=(37.193497,36.200088), max_iter=5000, fmt_out='ENVI', max_shift=200)
    # CR = COREG(im_reference, im_target, path_out=destination, wp=(37.173773,36.210160), max_iter=25000, fmt_out='ENVI', max_shift=200) # 5x5
    # CR = COREG(im_reference, im_target, path_out=destination, wp=(37.174130,36.210142), max_iter=25000, fmt_out='ENVI', max_shift=200) # 5x5
    try:
        CR.calculate_spatial_shifts()
        CR.correct_shifts()
    except:
        print("An exception occurred:" ,image)
        failures.append(image)
    print(image)

for image in images[2:]:
    correct(image)

for image in images[:1]:
    correct(image)

for image in failures:
    complete = False
    for source in images:
        im_reference    = f'{os.getcwd()}/{source}'
        im_target    = f'{images_dir}/{image}'
        destination  = f'{os.getcwd()}/{image}'.split(".tif")[0]+".bsq"
        if source != image and not complete:
            try:
                print("Trying with:", source)
                if os.path.exists(destination):
                    os.remove(destination)
                # CR = COREG(im_reference, im_target, path_out=destination, wp=(37.193485,36.199957), max_iter=1000, fmt_out='ENVI', max_shift=500)
                CR = COREG(im_reference, im_target, path_out=destination, wp=(37.193497,36.200088), max_iter=1000, fmt_out='ENVI', max_shift=500)
                CR.calculate_spatial_shifts()
                CR.correct_shifts()
                complete=True
            except:
                print(f"Couldn't do it with {source}")
        elif complete == True:
            print(f"Already completed: discarding {source}")
        else:
            print("No luck brother!")

print(failures)