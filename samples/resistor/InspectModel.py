#Runs a trained model, evaluating an image from val then saving it as a .jpg and saves

import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from samples.resistor import Resistor

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
RESISTOR_WEIGHTS_PATH = os.path.join(MODEL_DIR, "resistor20210510T2002/mask_rcnn_resistor_0029.h5") # TODO: update this path

config = Resistor.ResistorConfig()
RESISTOR_DIR = os.path.join(ROOT_DIR, "datasets\\resistor")

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# Load validation dataset
dataset = Resistor.ResistorDataset()
dataset.load_resistor(RESISTOR_DIR, "val")
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

#LOAD MODEL
#TODO - update path
weights_path = os.path.join(MODEL_DIR, "resistor20210510T2002/mask_rcnn_resistor_0029.h5") #Choose which model to load

#Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

#RUN DETECTION
image_id = dataset.image_ids[1] #random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]
print(r)
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
#Save the mask overlaid on the image
plt.savefig('7.jpg',bbox_inches='tight', pad_inches=-0.5,orientation= 'landscape') #TODO: Change file name

mask = r['masks']
mask = mask.astype(int)

print("==========Mask Shape==========")
print(mask.shape)

print("==========Image Shape==========")
print(image.shape)

#Extract the masks
for i in range(mask.shape[2]):
    temp = image.copy()
    for j in range(temp.shape[2]):
        temp[:,:,j] = temp[:,:,j] * mask[:,:,i]
    plt.figure(figsize=(8,8))
    #Save the masked image for each mask
    plt.imshow(temp)
    plt.savefig(f'7-mask{i}.jpg',bbox_inches='tight', pad_inches=-0.5,orientation= 'landscape') #TODO: Change file name

plt.close()

log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)