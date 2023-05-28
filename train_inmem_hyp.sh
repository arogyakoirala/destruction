#!/bin/bash
# EXERCISE_ENVIRONMENT="env"
# eval "$(conda shell.bash hook)"

source env/bin/activate

declare -a Cities=("aleppo" "aleppo,damascus,hama,homs,idlib,raqqa,deir-ez-zor,daraa" "aleppo,damascus" "damascus") 
declare -a BatchSizes=("32")
declare -a Filters=("32")
declare -a Units=("64" "128")
declare -a Dropouts=("0.05" "0.15")
declare -a LearningRates=("0.00003")

for city in "${Cities[@]}"; do
    for dropout in "${Dropouts[@]}"; do
        for batchsize in "${BatchSizes[@]}"; do
            for lr in "${LearningRates[@]}"; do
                for units in "${Units[@]}"; do
                    for filters in "${Filters[@]}"; do
                        # python -u train_inmem.py --cities $city --model double --filters $filters --units $units --lr $lr --batch_size $batchsize --dropout $dropout --data_dir ../data/destr_data --output_dir ../data/destr_outputs
                        python -u train_inmem.py --cities $city --model double --filters $filters --units $units --lr $lr --batch_size $batchsize --dropout $dropout --data_dir /lustre/ific.uv.es/ml/iae091/data --output_dir /lustre/ific.uv.es/ml/iae091/outputs
                    done
                done
            done
        done
    done
done
