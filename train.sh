#!/bin/bash
# EXERCISE_ENVIRONMENT="env"
# eval "$(conda shell.bash hook)"
# conda activate $EXERCISE_ENVIRONMENT

declare -a Dropouts=("0.05" "0.10" "0.15")
declare -a Units=("64" "128" "256")
declare -a Filters=("32" "64" "128")
declare -a LearningRates=("0.00001" "0.00003" "0.0001" "0.0003")



for dropout in "${Dropouts[@]}"; do
    for lr in "${LearningRates[@]}"; do
        for filter in "${Filters[@]}"; do
            for unit in "${Units[@]}"; do
                # python -u train.py --model double --dropout $dropout --lr $lr --filters $filter --units $unit --data_dir ../data/destr_data --output_dir ../data/destr_outputs
                python -u train.py --model double --dropout $dropout --lr $lr --filters $filter --units $unit --data_dir /lustre/ific.uv.es/ml/iae091/data --output_dir /lustre/ific.uv.es/ml/iae091/outputs 
            done
        done
    done
done