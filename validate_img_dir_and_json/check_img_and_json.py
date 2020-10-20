import os
import json


def read_json(json_file_path):
	json_cont = None
	with open(json_file_path, 'r') as f:
		json_cont = json.load(f)
	return json_cont



def check_diff(json_cont, img_dir):
    file_key_list = list(json_cont.keys())


    json_img_name = []

    for file_key in file_key_list:
        image_file_name = json_cont[file_key]['filename']
        #print(image_file_name)
        json_img_name.append(image_file_name)

    print("The number of image name in the annotated json is {}.\n".format(len(json_img_name)))
    set1_json = set(json_img_name)

    #print(set1)

    img = []

    for files in os.listdir(img_dir):
        if files.startswith('.') is False and files.endswith('json') is False:
            img.append(files)
            # print(files)

    print("The number of images in the image directory is {}.\n".format(len(img)))
    set2_img = set(img)
    diff_in_set1_json_set2_img = set1_json - set2_img
    if len(diff_in_set1_json_set2_img) == 0:
        print("The number of annotated images in the json is the same as the number of images in the image directory.\n")
    else:
        print("The are {} more annotations in the JSON than the number of images in the image directory.".format(len(diff_in_set1_json_set2_img)))
        print("\nThe extra annotations are: \n{}\n".format(diff_in_set1_json_set2_img))

    diff_in_set2_img_set1_json = set2_img - set1_json
    if len(diff_in_set2_img_set1_json) == 0:
        print("The number of images in the image directory is the same as the number of annotated images in the JSON file.\n")
    else:
        print("{} images present in the image directory is not in the annotated JSON.".format(len(diff_in_set2_img_set1_json)))
        print("\nThe extra images are: \n{}\n".format(diff_in_set2_img_set1_json))

if __name__ == '__main__':

    """
    IMG_SOURCE_DIR = folder that store all batches of images and json file
    """

    IMG_SOURCE_DIR = '/Users/johnathontoh/Desktop/python_files/dataset/farm_dams/image_source'

    for folder in os.listdir(IMG_SOURCE_DIR):
        if folder.startswith('.') is False:
            folder_path = os.path.join(IMG_SOURCE_DIR, folder)
            for files in os.listdir(folder_path):
                if files.startswith('.') is False and files.endswith('.json') is True:
                    print("--------------------------\n")
                    print("Directory: {}\n".format(folder_path))
                    json_file = read_json(os.path.join(folder_path, files))
                    check_diff(json_file, folder_path)






