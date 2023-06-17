import argparse
import os
import re
parser = argparse.ArgumentParser()
parser.add_argument("run_id", help="Model Run ID for which we want to generate predictions")
parser.add_argument("--cities", help="Pre File")
parser.add_argument("--data_dir", help="Model Run ID for which we want to generate predictions")
parser.add_argument("--output_dir", help="Model Run ID for which we want to generate predictions")
args = parser.parse_args()

"""
USAGE:
python -m predict 3 aleppo,daraa
"""


## For local
# CITIES = ['aleppo', 'daraa']
# OUTPUT_DIR = "../data/destr_outputs"
# DATA_DIR = "../data/destr_data"

## For artemisa
CITIES = ['aleppo', 'damascus', 'daraa', 'deir-ez-zor','hama', 'homs', 'idlib', 'raqqa']
OUTPUT_DIR = "/lustre/ific.uv.es/ml/iae091/outputs"
DATA_DIR = "/lustre/ific.uv.es/ml/iae091/data"

## For workstation
# CITIES = ['aleppo', 'damascus', 'daraa', 'deir-ez-zor','hama', 'homs', 'idlib', 'raqqa']
# OUTPUT_DIR = "../outputs"
# DATA_DIR = "../data"


if args.data_dir:
    OUTPUT_DIR = args.output_dir

if args.output_dir:
    DATA_DIR = args.data_dir

if args.cities:
    CITIES = [el.strip() for el in args.cities.split(",")]


RUN_DIR = f'{OUTPUT_DIR}/{args.run_id}'




def search_data(pattern:str='.*', directory:str='../data') -> list:
    '''Sorted list of files in a directory matching a regular expression'''
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    if len(files) == 1: files = files
    return files

for city in CITIES:

    if os.path.exists(f"{RUN_DIR}/predictions_{city}.csv"):
        print("File already exists!")
        os.remove(f"{RUN_DIR}/predictions_{city}.csv")

    pre_images  = search_data(pattern='^.*tif', directory=f'{DATA_DIR}/{city}/images/pre')
    post_images  = search_data(pattern='^.*tif', directory=f'{DATA_DIR}/{city}/images/post')


    for pre_ in pre_images:
        for post_ in post_images:
            print(city, "-", pre_.split("/")[-1], post_.split("/")[-1])

            os.system(f"python -m predict_chunk {args.run_id} {pre_} {post_} --data_dir {DATA_DIR} --output_dir {OUTPUT_DIR}")

