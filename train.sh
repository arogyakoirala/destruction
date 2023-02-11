#!/bin/bash
EXERCISE_ENVIRONMENT="env"
eval "$(conda shell.bash hook)"
conda activate $EXERCISE_ENVIRONMENT

python3 -u train.py --cities aleppo,damascus,daraa,deir-ez-zor,hama,homs,idlib,raqqa --model double --output_dir /lustre/ific.uv.es/ml/iae091