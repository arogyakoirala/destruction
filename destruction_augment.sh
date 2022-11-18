#!/bin/bash
EXERCISE_ENVIRONMENT="destruction"
eval "$(conda shell.bash hook)"
conda activate $EXERCISE_ENVIRONMENT
python3 destruction-augment.py --balance
