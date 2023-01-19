# Monitoring war destruction from space using machine learning (continued)

Extracting information on war related destruction is difficult because it relies on eyewitness accounts and manual detection on the field, which is not only costly for institutions carrying out these efforts, but also unsafe for the individuals carrying out this task. The information gathered is also incomplete, which makes it difficult for use in media reporting, carrying out humanitarian relief, understanding human rights violations, or academic reporting. This seminar introduces an automated approach to measure destruction in war damaged buildings by applying deep learning in publicly available satellite imagery. We adapt different neural network architectures and make them applicable for the building damage detection use case. As a proof of concept, we apply this method to the Syrian Civil War to reconstruct war destruction in these regions over time.

This is a follow-up project using Mueller et al. Monitoring war destruction from space using machine learning. Proceedings of the National Academy of Sciences, 118(23), 2021.

# Using this repository
This sections contains step by step instructions on how to get the code running on your system. 

* [1. High-level folder structure](#1-high-level-folder-structure)
* [2. The data directory.](#2-the-data-directory)
* [3. The destruction directory.](#3-the-destruction-directory)
    * [Environment setup](#environment-setup)
        * [Step 0: Create a virtual environment (one-time only)](#step-0-create-a-virtual-environment-one-time-only)
        * [Step 1: Activate the virtual environment:](#step-1-activate-the-virtual-environment)
        * [Step 2 (optional) Check if virtual environment is ready:](#step-2-optional-check-if-virtual-environment-is-ready)
* [4. Data preprocessing.](#4-data-preprocessing)
    * [One shot setup](#one-shot-setup)
    * [Step-by-step setup](#step-by-step-setup)
* [5. Model training and optimization.](#5-model-training-and-optimization)
    * [Examples](#examples)
    * [A note on managing different runs](#a-note-on-managing-different-runs)
* [6. Generating dense predictions.](#6-generating-dense-predictions)

In this document, we're assuming that satellite imagery for different cities has already been downloaded and is ready to use.

A note on satellite imagery format: Make sure that,

* The raster dimensions (height and width) are multiples of 256.
* The raster dimensions across dates are exactly the same.
* The raster is formatted as 8-bits unsigned integers.
* The raster is named city/image_YYYY_MM_DD

## 1. High-level folder structure

Have a root directory to host all the project files. In any location in your computer:

```
mkdir mwd
```

The `mwd` directory is your project root, this is where we are going to organize our data and code. At the high level it should contain the following subdirectories

```
|---- mwd
|-------- data
|------------ aleppo
|------------ damascus
|------------ ... more
|-------- destruction (will be downloaded from github)

```


## 2. The data directory.

For the code to run, the data directory must have:

1. Subdirectories for each city named after the city.
2. Each city level subdirectory must contain:

    - A folder called `images`, which contains:
        - A subdirectory called `pre`, with pre images named using this convention: image_{yyyy_mm_dd}.tif
        - A subdirectory called `post`, with pre images named using this convention: image_{yyyy_mm_dd}.tif     
    - A folder called `labels`, which is empty.
    - A folder called `other`, which contains: 
        - Damage annotation shapefile: {city}_damage.gpkg
        - Settlement area shapefile: {city}_settlement.gpkg
        - No analysis zone shapefile: {city}_noanalysis.gpkg

Here's what the structure looks like for the city of aleppo:

```
|---- mwd /
|-------- data /
|------------ aleppo /
|---------------- images /
|-------------------- pre /
|------------------------ image_2011_01_01.tif
|------------------------ image_2011_06_01.tif
|------------------------ ...more images
|-------------------- post /
|------------------------ image_2015_01_28.tif
|------------------------ image_2017_06_01.tif
|------------------------ image_2017_06_01.tif
|------------------------ ...more images
|---------------- labels /
|---------------- others /
|-------------------- aleppo_damage.gpkg
|-------------------- aleppo_settlement.gpkg
|-------------------- aleppo_noanalysis.gpkg
|------------ damascus /
|------------ daraa /
|------------ ...more cities

```

## 3. The destruction directory.

The `mwd/destruction` directory will contain code present in this repo. To set it up:

```sh
# Go to project root
cd mwd

# Clone this repository:
git clone https://github.com/arogyakoirala/destruction.git
```

This is the folder we run code from. 

### Environment setup

This code requires certain libraries, and we need to make sure these are installed. This section describes how you can do that. 



#### Step 0: Create a virtual environment (one-time only)
```
cd mwd
venv destr
```

#### Step 1: Activate the virtual environment:

```
cd mwd
source destr/bin/activate
```

#### Step 2 (optional) Check if virtual environment is ready:

If the following command executes successfully without any errors, its safe to assume that our environment is ready.

```
cd mwd
cd destruction
python checkenv.py
```

**IMPORTANT**: All of the python code must be run from the `mwd/destruction/` folder. This is true for all examples shown from here on. I will assume that you're in the `/mwd/destruction/` directory and skip all the `cd` statements going forward. 


## 4. Data Preprocessing

At a high level the following steps are carried out in the setup step for each city:

1. Isolate analysis zones
2. Create master sampling raster, which represents how our analysis zone will be split into test, train, and validation set at the patch label.
3. Perform label augmentation at the patch level.
4. Generate tiles.
5. Balance the training set data (50-50 split between positive and negative samples)
5. Shuffle training set data. 
5. Shuffle agin, just to be safe. 


### One shot setup

Proivided the data directory is ready according to the guidelines above, a master shell script allows us to carry all of these steps in one shot. To do this, [make sure the python environment is loaded](#environment-setup), and then run the following commands:

```sh
cd mwd
cd destruction
chmod +x prep_data.sh
./prep_data.sh
```
If you want more flexibility, follow the instruction on the step-by-step setup section below. 

We are now ready to proceed with model training and optimization.

### Step-by-step setup



In the setup step, for each city:

1. We first isolate the analysis region, by using the shapefiles for settlement regfons and no-analysis zone. We also use an image (any image) from the city to crop the analysis region into the image bounds.
2. We then divide this analysis region into patches of 128 pixels, and randomly assign patches to the train, test, and validation set. This is the sampling step. We store thijs information in a raster file called {city}_sample.tif inside the `data/{city}/others/` folder

> Steps one and two are automatically carried out my running sample.py and specifying the city as a parameter. [Ensure that the python environment is loaded](#environment-setup) and then run: `python sample.py --city aleppo` 

3. We then proceed to label augmentation at the patch level. This is done using the damage annotations shapefiles and the sampling raster. The result will be patch-level rasters for each imagery date, which will be stored in the `/data/{city}/labels` folder.

> For label augmentation, [ensure that the python environment is loaded](#environment-setup) and then run: `python label.py --city aleppo`. 
>
> **Check:** *To check if this worked correctly, you can open any label raster (from) `/data/{city}/labels` folder in QGIS and overlay the damage annotation shapefile. TUse visual inspection to validate if label assignment has been done correctly.*

4. Now that we have the labels, and the images, we proceed to the tiling operation, where we generate numpy arrays for every patch-label combination, and store it in the `/data/{city}/others/` directory as zarr files.

The following 9 zarr files will be generated:

```
|---- mwd /
|-------- data /
|------------ aleppo /
|---------------- (...other folders..)
|---------------- others /
|-------------------- aleppo_im_tr_pre.zarr (pre images; training set)
|-------------------- aleppo_im_tr_post.zarr (post images; training set)
|-------------------- aleppo_la_tr.zarr (labels; training set)
|-------------------- aleppo_im_va_pre.zarr (pre images; validation set)
|-------------------- aleppo_im_va_post.zarr (post images; validation set)
|-------------------- aleppo_la_va.zarr (labels; validation set)
|-------------------- aleppo_im_te_pre.zarr (pre images; test set)
|-------------------- aleppo_im_te_post.zarr (post images; test set)
|-------------------- aleppo_la_te.zarr (labels; test set)
```

> To run the tiling step, [ensure that the python environment is loaded](#environment-setup) and then run: `python tile.py --city aleppo` 

5. The next step is balancing the data, where we append randomly selected positive examples (with replacement) to the zarr files so there's a 50-50 split between positive and negative examples:

> To run the balancing step, [ensure that the python environment is loaded](#environment-setup) and then run: `python balance.py --city aleppo`

6. The final step is to shuffle the data. 

> To run the shuffling step, [ensure that the python environment is loaded](#environment-setup) and then run: `python shuffle.py --city aleppo`

It is advisable to run the shuffling step twice to ensure the data is properly shuffled.

We are now ready to proceed with model training and optimization.

## 5. Model training and optimization
The model training code is designed in a way to facilitate many runs (provided there's enough memory in the machine to do so). For each model run, you can specify:

1. The cities from which to use data for the model training step.
2. The kind of model you want to run:
    1. Siamese Network
    2. Double convolution Network 


### Examples
1. To train double convolution network using data from aleppo and raqqa, , [ensure that the python environment is loaded](#environment-setup) and then run: `python train.py --cities aleppo,raqqa --model double`


2. To train siamese network using data from aleppo, raqqa, damascus, homs, [ensure that the python environment is loaded](#environment-setup) and then run:`
python train.py --cities aleppo,raqqa,damascus,homs --model snn
`

3. To train double convolution network using data from aleppo only, [ensure that the python environment is loaded](#environment-setup) and then run:`python train.py --cities aleppo --model snn`

#### Run as background process:

To run in background, [ensure that the python environment is loaded](#environment-setup) and then use nohup: `nohup python -u train.py --cities aleppo,raqqa --model double > logfile.out &`

This runs the command in the background and logs out all the ouput to `mwd/destruction/logfile.out`. Make sure the logfile doesnt exist prior to this step, or use a differen name everytime. To monitor progress, you can simply review the last line on logfile.out


#### A note on managing different runs
The training step assigns a RUN_ID for every run, and creates a folder called `mwd/outputs/{code}` where it will store temporary files required for, and final results after the execution. The RUN_ID is srinted to standard output every time you run train.py, so make sure you make a note of that.

Theoretically this setup allows us to run multiple training steps in parallel using nohup. 

```sh
nohup python -u train.py --cities aleppo,raqqa --model double > logfile_ar.out &

nohup python -u train.py --cities aleppo --model double > logfile_aleppo.out &
```

What we're bottlenecked by, however, is the fact that for each RUN_ID, the /outputs/{RUN_ID} folder will contain a composite zarr file, that contains training data for the cities involved in the model run. This causes a disk-space related bottleneck, which i'm currently thinking of a way around.




## 6. Generating dense predictions:

You will use the RUN_ID to specify what trained model you're generating predictions for. Currently, the code supports dense prediction generation at the city level.

To generate predictions, [ensure that the python environment is loaded](#environment-setup) and then run: `python dense_predict.py REPLACE_WITH_RUN_ID --cities aleppo,daraa`

The prediction results will be found in the `mwd/outputs/{RUN_ID}/predictions`

The ouputs are patch level raster files which contain the probability of damage for each patch based on the model. 



#### Run as background process:

To run in background, [ensure that the python environment is loaded](#environment-setup) and then use nohup: `nohup python -u dense_predict.py REPLACE_WITH_RUN_ID --cities aleppo,raqqa --model double > logfile_predict.out &`

This runs the command in the background and logs out all the ouput to `mwd/destruction/logfile_predict.out`. Make sure the logfile doesnt exist prior to this step, or use a differen name everytime. To monitor progress, you can simply review the last line on logfile.out
