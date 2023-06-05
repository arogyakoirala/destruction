#!/bin/bash
# source ../environments/destr/bin/activate
source env/bin/activate

declare -a Cities=("hama")
# declare -a Cities=("aleppo" "daraa")
# declare -a Cities=("raqqa")

declare -a data_dir=$1
echo "Data Dir: $data_dir";

for city in "${Cities[@]}"; do
    echo "Sampling:" $city
    python sample.py --city $city --data_dir $data_dir
    echo "Labeling:" $city
    python label.py --city $city --data_dir $data_dir
    echo "Tiling:" $city
    python tile.py --city $city --data_dir $data_dir
    echo "Balancing:" $city
    python balance.py --city $city --data_dir $data_dir
    echo "Shuffling:" $city
    python shuffle.py --city $city --data_dir $data_dir --block_size 10000
    echo "Shuffling again:" $city
    python shuffle.py --city $city --data_dir $data_dir --block_size 20000
done