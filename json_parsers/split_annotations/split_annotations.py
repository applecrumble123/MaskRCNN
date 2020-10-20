import json
import os
from shutil import copy
from json_parsers.split_annotations.config_split import *


def create_dir(dir_path):
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)


def read_json(json_file_path):
	json_cont = None
	with open(json_file_path, 'r') as f:
		json_cont = json.load(f)
	return json_cont

def copy_file(src_path, dest_path):
	copy(src_path, dest_path)


def create_image_split(annot_json, output_dir):
	file_key_list = list(annot_json.keys())
	if len(file_key_list) >0:
		for file_key in file_key_list:
			image_file_name = annot_json[file_key]['filename']
			image_src_file_path = os.path.join(INPUT_DIR, image_file_name)

			if os.path.exists(image_src_file_path):
				image_dest_file_path = os.path.join(output_dir, image_file_name)
				copy_file(image_src_file_path, image_dest_file_path)
			else:
				# if file doesn't exceed, we can remove key
				print('Image file {} doesn\'t exists'.format(image_src_file_path))
				del annot_json[file_key]
		return annot_json

def write_json(annot_json, output_path):
	with open(output_path, 'w') as outfile:
		json.dump(annot_json, outfile)

def split_json(annot_json):
	first_split_len = int(len(annot_json)*OUTPUT_SPLIT)
	annot_json_1 = dict(list(annot_json.items())[:first_split_len])
	annot_json_2 = dict(list(annot_json.items())[first_split_len:])
	return annot_json_1, annot_json_2


if __name__ == '__main__':

	if OUTPUT_SPLIT >1 or OUTPUT_SPLIT <0:
		print('Invalid split value. The value range is 0 to 1.')
		exit(0)


	create_dir(OUTPUT_DIR_1)
	create_dir(OUTPUT_DIR_2)

	parent_annot_json = read_json(os.path.join(INPUT_DIR, INPUT_DIR_JSON))

	annot_json_1, annot_json_2 = split_json(parent_annot_json)


	annot_json_1 = create_image_split(annot_json_1, OUTPUT_DIR_1)
	output_path_json_1 = os.path.join(OUTPUT_DIR_1, OUTPUT_DIR_1_JSON)
	write_json(annot_json_1, output_path_json_1)
	
	annot_json_2 = create_image_split(annot_json_2, OUTPUT_DIR_2)
	output_path_json_2 = os.path.join(OUTPUT_DIR_2, OUTPUT_DIR_2_JSON)
	write_json(annot_json_2, output_path_json_2)
	