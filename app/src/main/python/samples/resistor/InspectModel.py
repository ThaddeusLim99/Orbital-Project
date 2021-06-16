#Runs a trained model, evaluating an image from val or a unique image then saving it as a .jpg
import os
import sys
import random
import math
import re
import time
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io

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
#ESISTOR_WEIGHTS_PATH = os.path.join(MODEL_DIR, "resistor20210515T1841/mask_rcnn_resistor_0029.h5") # TODO: update this path

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

# Since the submission system does not permit overlapped masks, we have to fix them
def refine_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m]==True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois

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
weights_path = os.path.join(MODEL_DIR, "TrainedUsingBestModels\mask_rcnn_resistor_0044.h5") #Choose which model to load

#Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

#RUN DETECTION
image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))
'''
#completely new image
image_path = "C:\\Users\\Mloong\\Downloads\\resistor-public-domain-modified.jpg"

# Run object detection
#results = model.detect([img], verbose=1)
image = skimage.io.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
'''
results = model.detect([image])

# Display results
ax = get_ax(1)
r = results[0]

# print(r)
#print the coordinates of the first bounding box, format (y1, x1, y2, x2)
#print (r['rois'][0])

if r['masks'].size > 0:
    masks = np.zeros((image.shape[0], image.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
    for m in range(r['masks'].shape[-1]):
        masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 
                                    (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    y_scale = image.shape[0]/1024
    x_scale = image.shape[1]/1024
    rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)
    
    masks, rois = refine_masks(masks, rois)
else:
    masks, rois = r['masks'], r['rois']

visualize.display_instances(image, rois, masks, r['class_ids'], 
                            ['bg'] + dataset.class_names, r['scores'],
                            ax=ax, title="Predictions")
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax, title="Predictions")

#Save the mask overlaid on the image
name = "11"
plt.savefig(f"{name}.jpg",bbox_inches='tight', pad_inches=-0.5,orientation= 'landscape') #TODO: Change file name

mask = r['masks']
mask = mask.astype(int)

print("==========Mask Shape==========")
print(mask.shape)

print("==========Image Shape==========")
print(image.shape)

#Extract the masks
'''
for i in range(mask.shape[2]):
    temp = image.copy()
    for j in range(temp.shape[2]):
        temp[:,:,j] = temp[:,:,j] * mask[:,:,i]
    plt.figure(figsize=(8,8))
    #Save the masked image for each mask
    plt.imshow(temp)
    plt.savefig(f'{name}-mask{i}.jpg',bbox_inches='tight', pad_inches=-0.5,orientation= 'landscape') #TODO: Change file name
'''
plt.close()

log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)