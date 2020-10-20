import os
import json
import numpy as np
from shutil import copy
from filter_large_farm_dams.config_large_farm_dams import *

def get_text_file():
    with open(LARGE_FARM_DAM_TEXTFILE, "r") as f:
        large_farm_dams_img_name = [line.rstrip('\n') for line in f]
    return large_farm_dams_img_name

def get_json_file():
    with open(MERGED_JSON_FILE_PATH, 'r') as f:
        merged_json_file = json.load(f)
    return merged_json_file


def process_img(large_farm_dams_img_name):
    for img_name in large_farm_dams_img_name:
        file_path = os.path.join(MERGED_IMAGE_PATH, img_name)
        image_dest_file_path = os.path.join(LARGE_FARM_DAM_DIR, img_name)
        copy(file_path, image_dest_file_path)


def process_json(large_farm_dams_img_name, json_file):
    file_key_list = list(json_file.keys())
    for file_key in file_key_list:
        image_file_name = json_file[file_key]['filename']
        if image_file_name not in large_farm_dams_img_name:
            del json_file[file_key]
    return json_file



if __name__ == '__main__':

    large_farm_dams_img_name = get_text_file()

    merged_json_file = get_json_file()

    process_img(large_farm_dams_img_name)

    processed_json_file = process_json(large_farm_dams_img_name, merged_json_file)

    with open(os.path.join(LARGE_FARM_DAM_DIR, OUTPUT_JSON_FILE_NAME), 'w') as outfile:
        json.dump(processed_json_file, outfile)