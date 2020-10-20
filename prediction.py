import os
import sys
import warnings

import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

from utils.loader import InferenceConfig, FarmDamDataset
import mrcnn.model as modellib

from mrcnn.config import Config
from mrcnn.model import log
from mrcnn import visualize
from mrcnn import utils
import tensorflow as tf
from mrcnn.utils import compute_matches

import argparse
import textwrap
import shutil

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=textwrap.dedent('''\
------------------------------------------------------------------------------------
Please set your ROOT_DIR, SAVED_MODEL_DIR, COCO_MODEL_PATH, DATASET_PATH in the prediction2.py accordingly.
------------------------------------------------------------------------------------

    ROOTDIR --> Root folder for Mask RCNN

    SAVED_MODEL_DIR --> Directory Path to save your newly trained weights
    
    SAVED_MODEL_PATH --> File path of your newly trained weights

    COCO_MODEL_PATH --> File path of your coco dataset weights 

    DATASET_PATH --> Directory path to your train, test and val folders
    
    TEST_IMAGE_FOLDER --> Directory path to your test images

        '''))

# add the arguments u want the user to input in the command line
parser.add_argument('-train', '--train_json', type=str, metavar='', help='File name of json file for the train set')
parser.add_argument('-val', '--val_json', type=str, metavar='', help='File name of json file for the val set')
parser.add_argument('-test', '--test_json', type=str, metavar='', help='File name of json file for the test set')
parser.add_argument('-i', '--test_image', type=str, metavar='', help='File name of test image')

args = parser.parse_args()


ROOT_DIR = "/Users/johnathontoh/Desktop/python_files"

# path where the model is saved
SAVED_MODEL_DIR = os.path.join(ROOT_DIR, "saved_model")

# path to pretrained coco model
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "initial_weights", "mask_rcnn_coco.h5")

# dataset
DATASET_PATH = os.path.join(ROOT_DIR, "dataset/farm_dams")

#SAVED_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "mask_rcnn_satellite_farm_dams.h5")

TEST_IMAGE_FOLDER = os.path.join(ROOT_DIR, 'dataset/farm_dams/test')

inference_config = InferenceConfig()

def prepare_train_val(train_json, val_json, test_json):
    dataset_train = FarmDamDataset()
    dataset_train.load_dam(DATASET_PATH, "train", train_json)
    dataset_train.prepare()

    # validation images dir
    dataset_val = FarmDamDataset()
    dataset_val.load_dam(DATASET_PATH, "val", val_json)
    dataset_val.prepare()

    # test images dir
    dataset_test = FarmDamDataset()
    dataset_test.load_dam(DATASET_PATH, "test", test_json)
    dataset_test.prepare()

    return dataset_train, dataset_val, dataset_test

def load_transfered_learned_model(model_per_dir, config):
  # Create model in training mode
  model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_per_dir)
  # Load weights trained on MS COCO, but skip layers that
  # are different due to the different number of classes
  for i in os.listdir(SAVED_MODEL_DIR):
      if i.startswith('.') is False:
          if len(os.listdir(os.path.join(SAVED_MODEL_DIR, i))) == 0:  # Check is empty..
              shutil.rmtree(os.path.join(SAVED_MODEL_DIR, i))

  print("Model path: {}".format(model.find_last()))
  # find the last weight in the directory
  model.load_weights(model.find_last(), by_name=True)
  #print("Loading weights from ", model_saved_path)
  #model.load_weights(model_saved_path, by_name=True)
  return model

def load_image(image_path):
  img = cv2.imread(image_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (inference_config.IMAGE_MIN_DIM, inference_config.IMAGE_MIN_DIM))
  return img

model = load_transfered_learned_model(SAVED_MODEL_DIR, inference_config)


def infer_object(image):
  # making prediction
  results = model.detect([image], verbose=1)

  results = results[0]
  visualize.display_instances(image, results['rois'], results['masks'], results['class_ids'],
                              dataset_val.class_names, results['scores'], figsize=(8, 8))

  print("\nThere are {} farm dams detected.".format(results['masks'].shape[-1]))
  for i in range(results['masks'].shape[-1]):
    # get number of pixels for each mask
    mask = results['masks'][:, :, i]
    print(mask.sum())

  return results, visualize

def validate_test_img(dataset_test, test_image_name):
    image_ids = (dataset_test.image_ids)
    for image_id in range(len(image_ids)):
        if dataset_test.image_info[image_id]['id'] == test_image_name:
            # get the mask id
            print("image_id ", image_id, dataset_train.image_reference(image_id))

            # load the mask
            mask, class_ids = dataset_test.load_mask(image_id)

            # load the image
            image = dataset_test.load_image(image_id)

            # original shape
            original_shape = image.shape

            # resize the image with reference to the config file
            image, window, scale, padding, _ = utils.resize_image(
                image,
                min_dim=Config.IMAGE_MIN_DIM,
                max_dim=Config.IMAGE_MAX_DIM,
                mode=Config.IMAGE_RESIZE_MODE)
            mask = utils.resize_mask(mask, scale, padding)

            # compute the mask
            bbox = utils.extract_bboxes(mask)

            # Display image and additional stats
            print("image_id: ", image_id, dataset_train.image_reference(image_id))
            print("Original shape: ", original_shape)
            log("image", image)
            log("mask", mask)
            log("class_ids", class_ids)
            log("bbox", bbox)

            # display the original data
            visualize.display_instances(image, bbox, mask, class_ids, dataset_test.class_names, figsize=(8, 8))
            print("\nThere are {} actual dams for test image '{}'.".format(mask.shape[-1], test_image_name))
            for i in range(mask.shape[-1]):
                mask_pixel = mask[:,:, i]
                # get number of pixels for each mask
                print(mask_pixel.sum())

if __name__ == '__main__':

    inference_config.display()

    dataset_train, dataset_val, dataset_test = prepare_train_val(args.train_json, args.val_json, args.test_json)


    image_path = os.path.join(TEST_IMAGE_FOLDER, args.test_image)

    img = load_image(image_path)

    validate_test_img(dataset_test, args.test_image)

    print("")

    results = infer_object(img)







