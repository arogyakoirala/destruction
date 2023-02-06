#!/bin/bash
# Purpose: Read Comma Separated CSV File and save images
# Author: Vivek Gite under GPL v2.0+, modified by Arogya K
# ------------------------------------------

INPUT=links.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read folder name link
do
    dest_folder="${folder}/images/"
    dest="${dest_folder}/$name"
    mkdir -p $dest_folder

    if [ -e $dest ]
    then
        echo "File already exists: ${dest}"
    else
        wget $link -O $dest
    fi
done < $INPUT
IFS=$OLDIFS