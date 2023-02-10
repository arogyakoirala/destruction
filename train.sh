#!/bin/bash
EXERCISE_ENVIRONMENT="env"
eval "$(conda shell.bash hook)"
conda activate $EXERCISE_ENVIRONMENT

python3 -u train.py --cities $1 --model double --output_dir /lustre/ific.uv.es/ml/iae091