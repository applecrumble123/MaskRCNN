import os
import sys
import random
import math
import re
import time
import warnings

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import textwrap
from imgaug import augmenters as iaa

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from utils.loader import DamConfig, FarmDamDataset
import argparse
import shutil

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=textwrap.dedent('''\
------------------------------------------------------------------------------------
Please set your ROOT_DIR, SAVED_MODEL_DIR, COCO_MODEL_PATH, DATASET_PATH in the train2.py accordingly.
------------------------------------------------------------------------------------

    ROOTDIR --> Root folder for Mask RCNN
    
    SAVED_MODEL_DIR --> Directory Path to save your newly trained weights
    
    COCO_MODEL_PATH --> File path of your coco dataset weights 
    
    DATASET_PATH --> Directory path to your train, test and val folders

        '''))

# add the arguments u want the user to input in the command line


#mrcnn_path = '/Users/johnathontoh/Desktop/python_files/mrcnn'
#sys.path.append(mrcnn_path)

# Root directory of the project
ROOT_DIR = "/Users/johnathontoh/Desktop/python_files"


# path where the model is saved
SAVED_MODEL_DIR = os.path.join(ROOT_DIR, "saved_model")

# path to pretrained coco model
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "initial_weights", "mask_rcnn_coco.h5")


# dataset
DATASET_PATH = os.path.join(ROOT_DIR, "dataset/farm_dams")



if __name__ == '__main__':

    parser.add_argument('-t', '--train_json', type=str, metavar='', help='File name of json file for the train set')
    parser.add_argument('-v', '--val_json', type=str, metavar='', help='File name of json file for the val set')
    parser.add_argument('-e', '--epoch', type=int, metavar='', help='Number of epoch for the model to run')
    parser.add_argument('-l', '--layers', type=str, metavar='',
                        help='Passing "heads" freezes all layers except the head. Passing "all" runs all the layer')
    parser.add_argument('-init', '--initialise', type=str, metavar='',
                        help='Enter "coco" to initialise with coco weights. Enter "last" to initialise with last trained weights.')

    args = parser.parse_args()

    def prepare_train_val(train_json, val_json):
        dataset_train = FarmDamDataset()
        dataset_train.load_dam(DATASET_PATH, "train", train_json)
        dataset_train.prepare()

        # validation images dir
        dataset_val = FarmDamDataset()
        dataset_val.load_dam(DATASET_PATH, "val", val_json)
        dataset_val.prepare()

        return dataset_train, dataset_val


    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=45),
                   iaa.Affine(rotate=90),
                   iaa.Affine(rotate=135),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])


    def get_pretrained_model(pretrained_model_weights, model_directory, initialise):
        init_with = initialise
        # Create model in training mode
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=model_directory)
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        if init_with == "coco":
            model.load_weights(pretrained_model_weights, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            # train from last saved epoch
            # remove any empty directory when running "model = modellib.MaskRCNN(mode="training", config=config, model_dir=model_directory)"
            # the directory is created based on the current time, so an empty directory is always created
            for i in os.listdir(SAVED_MODEL_DIR):
                if i.startswith('.') is False:
                    if len(os.listdir(os.path.join(SAVED_MODEL_DIR, i))) == 0:  # Check is empty..
                        shutil.rmtree(os.path.join(SAVED_MODEL_DIR, i))

            # find the last weight in the directory
            model.load_weights(model.find_last(), by_name=True)
        return model


    dataset_train, dataset_val = prepare_train_val(args.train_json, args.val_json)

    config = DamConfig()
    config.display()


    model = get_pretrained_model(COCO_MODEL_PATH, SAVED_MODEL_DIR, args.initialise)
    # Train the head branches.


    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=args.epoch,
                layers=args.layers,
                #augmentation=augmentation
                )



