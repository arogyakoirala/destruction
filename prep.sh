#!/bin/bash
EXERCISE_ENVIRONMENT="env"
eval "$(conda shell.bash hook)"
conda activate $EXERCISE_ENVIRONMENT
# ./prep_data.sh
# python3 -u destruction-augment.py
# python3 -u destruction-ready-data-snn.py --balance --pre_image_index 0,1
# python3 -u destruction-optimize.py 



declare -a Cities=("aleppo" "damascus" "daraa" "deir-ez-zor" "hama" "homs" "idlib" "raqqa")
# declare -a Cities=("aleppo" "daraa")
# declare -a Cities=("raqqa")

declare -a data_dir=$1
echo "Data Dir: $data_dir";

for city in "${Cities[@]}"; do
    echo "Sampling:" $city
    python -u sample.py --city $city --data_dir $data_dir
    echo "Labeling:" $city
    python -u label.py --city $city --data_dir $data_dir
    echo "Tiling:" $city
    python -u tile.py --city $city --data_dir $data_dir
    echo "Balancing:" $city
    python -u balance.py --city $city --data_dir $data_dir
    echo "Shuffling:" $city
    python -u shuffle.py --city $city --data_dir $data_dir
    echo "Shuffling again:" $city
    python -u shuffle.py --city $city --data_dir $data_dir
done