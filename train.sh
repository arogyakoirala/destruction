#!/bin/bash
# EXERCISE_ENVIRONMENT="env"
# eval "$(conda shell.bash hook)"
# conda activate $EXERCISE_ENVIRONMENT

python -u train --cities aleppo,daraa,hama --model double --data_dir /lustre/ific.uv.es/ml/iae091/data --output_dir /lustre/ific.uv.es/ml/iae091/outputs