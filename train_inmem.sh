#!/bin/bash
EXERCISE_ENVIRONMENT="env"
eval "$(conda shell.bash hook)"
conda activate $EXERCISE_ENVIRONMENT

python -u train_inmem.py --cities aleppo --model double --data_dir /lustre/ific.uv.es/ml/iae091/data --output_dir /lustre/ific.uv.es/ml/iae091/outputs