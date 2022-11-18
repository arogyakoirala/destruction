#!/bin/bash
EXERCISE_ENVIRONMENT="destruction"
eval "$(conda shell.bash hook)"
conda activate $EXERCISE_ENVIRONMENT
python3 destruction-augment.py
python3 destruction-ready-data-snn.py --balance --pre_image_index 0,1
python3 destruction-optimize.py 
