#!/bin/bash
EXERCISE_ENVIRONMENT="env"
eval "$(conda shell.bash hook)"
conda activate $EXERCISE_ENVIRONMENT
./prep_data.sh
# python3 -u destruction-augment.py
# python3 -u destruction-ready-data-snn.py --balance --pre_image_index 0,1
# python3 -u destruction-optimize.py 
