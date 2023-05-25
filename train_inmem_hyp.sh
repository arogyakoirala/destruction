#!/bin/bash
EXERCISE_ENVIRONMENT="env"
eval "$(conda shell.bash hook)"
conda activate $EXERCISE_ENVIRONMENT

declare -a Cities=("aleppo" "aleppo,damascus,hama,homs,idlib,raqqa,deir-ez-zor,daraa" "aleppo,damascus" "damascus") 
declare -a BatchSizes=("128" "64" "32")
declare -a Filters=("8" "16" "32")
declare -a Units=("32" "64" "128")
declare -a Dropouts=("0.15" "0.25" "0.35")
declare -a LearningRates=("0.3" "0.03" "0.003")

for city in "${Cities[@]}"; do
    for dropouts in "${Dropouts[@]}"; do
        for batchsize in "${BatchSizes[@]}"; do
            for lr in "${LearningRates[@]}"; do
                for units in "${Units[@]}"; do
                    for filters in "${Filters[@]}"; do
                        python -u train_inmem_hyp.py --cities $city --model double --filters $filters --units $units --lr $lr --batch_size $batch_size --dropout $dropout --city $city --data_dir /lustre/ific.uv.es/ml/iae091/data --output_dir /lustre/ific.uv.es/ml/iae091/outputs
                    done
                done
            done
        done
    done
done
