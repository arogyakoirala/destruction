#!/bin/bash
EXERCISE_ENVIRONMENT="destruction"
eval "$(conda shell.bash hook)"
conda activate $EXERCISE_ENVIRONMENT
python3 -u destruction-augment.py
python3 -u destruction-ready-data-snn.py --balance --pre_image_index 0,1
python3 -u destruction-optimize.py 
