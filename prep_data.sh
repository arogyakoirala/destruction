#!/bin/bash
source /usr/local/anaconda3/condabin/conda
conda activate des-linux

declare -a Cities=("aleppo" "damascus" "daraa" "deir-ez-zor" "hama" "homs" "idlib" "raqqa")
# declare -a Cities=("raqqa")

for city in "${Cities[@]}"; do
    echo "Sampling:" $city
    python sample.py --city $city
    echo "Labeling:" $city
    python label.py --city $city
    echo "Tiling:" $city
    python tile.py --city $city
    echo "Balancing:" $city
    python balance.py --city $city
    echo "Shuffling:" $city
    python shuffle.py --city $city
    echo "Shuffling again:" $city
    python shuffle.py --city $city
done