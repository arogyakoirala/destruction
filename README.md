# destruction

All of the code is present in the experiments/ directory.

### Directory structure

```
|---- mwd
    |---- data
        |------ aleppo_cropped
            |---- images
                |---- tif files of images
            |---- labels
                |---- label rasters (generated from code)
            |---- others
                |---- annotations
                |---- no analysis zone shapefile
                |---- settlement zone shapefile
                |---- master sample split raster file (generated from code)
            |---- predictions
    |---- destruction (this repository)
```

## Workflow

### Sampling
```
cd experiments/sampling
python sampling.py
```

### Labeling
```
cd experiments/labeling
python labeling.py
```

### Patch Generation
```
cd experiments/tiling
python tiling.py
```

### Balancing
```
cd experiments/balance
python balance.py
```

### Shuffling
```
cd experiments/shuffle
python shuffle.py
```

### Generate Laplacian for patches
```
cd experiments/laplacian_patches
python laplacian_patches.py
```

### Generate HOGs for patches
```
cd experiments/hog_patches
python hog_patches.py
```

### Run SNN with raw image
```
cd experiments/snn_inmem
python snn_inmen.py --suffix im --bands 3
```

### Run SNN with laplacians image
```
cd experiments/snn_inmem
python snn_inmen.py --suffix lap --bands 1
```

### Run SNN with HOG Descriptors image
```
cd experiments/snn_inmem
python snn_inmen.py --suffix hog --bands 1
```
