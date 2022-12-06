#!/bin/bash
source /usr/local/anaconda3/condabin/conda
conda activate mlp

declare -a Cities=("aleppo" "raqqa" "damascus")

for city in "${Cities[@]}"; do
    python sample.py --city $city
    python label.py --city $city
    python tile.py --city $city
    python balance.py --city $city
    python shuffle.py --city $city
done