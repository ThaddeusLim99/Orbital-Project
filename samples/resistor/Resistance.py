# Runs a trained model, evaluating an image from val or a unique image then saving it as a .jpg
import os
import sys
import random
from time import time
import cv2
import numpy as np
import tensorflow as tf
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
from samples.resistor.ColourDetection import ColourSeparation

#include white balancing
from samples.resistor.WB_sRGB_Python.classes import WBsRGB as wb_srgb

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
#renamed DEFAULT_LOGS_DIR to MODEL_DIR
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

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

def run_detection(img, model, save=False):
    # Run object detection
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    t0 = time()

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

    if save:
        visualize.display_instances(image, rois, masks, r['class_ids'], 
                                    ['bg'] + ['resistor'], r['scores'], #since we are only detecting resistors
                                    ax=ax, title="Predictions")

        #Save the mask overlaid on the image
        plt.savefig("image_mask_overlay.png",bbox_inches='tight', pad_inches=-0.5, orientation='landscape')

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
        
        #Crop the image to only fit the bounding box
        print("==========Coordinates of Bounding Box in the form of [y1 x1 y2 x2]==========")
        print(r['rois'][i])
        y1, x1, y2, x2 = r['rois'][i]
        temp = temp[y1:y2, x1:x2]

        if save:
            # Saving the masked image
            cv2.imwrite(f'image-mask{i}.png', temp)
    plt.close()

    print("done in %0.3fs." % (time() - t0))

    return temp

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Extracting the masks of an Image, one for the mask overlaid over the image, and one for each individual masked image')
    parser.add_argument("-i", "--image", required=True,
                        metavar="path or URL to image",
                        help='Image to run the detection on')
    parser.add_argument("-wb", required=False, default=False,
                        help='Enable white balancing, set to True to enable')

    args = parser.parse_args()

    config = Resistor.ResistorConfig()

    class InferenceConfig(config.__class__):
    # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    
    config = InferenceConfig()
    config.display()

    #changed this part
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    WEIGHTS_PATH = os.path.join(MODEL_DIR, "mask_rcnn_resistor_0044.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(WEIGHTS_PATH):
        print("h5 file not found")

    # Load weights trained on MS-COCO
    print("Loading weights ", WEIGHTS_PATH)

    #changed this part
    model.load_weights(WEIGHTS_PATH, by_name=True)

    image_path = args.image
    image = cv2.imread(image_path)

    if args.wb:
        # Using https://github.com/mahmoudnafifi/WB_sRGB
        # use upgraded_model= 1 to load our new model that is upgraded with new
        # training examples.
        upgraded_model = 1
        # use gamut_mapping = 1 for scaling, 2 for clipping (our paper's results
        # reported using clipping). If the image is over-saturated, scaling is
        # recommended.
        gamut_mapping = 2

        # processing
        # create an instance of the WB model
        wbModel = wb_srgb.WBsRGB(gamut_mapping=gamut_mapping,
                                upgraded=upgraded_model)
        image = wbModel.correctImage(image) * 255  # white balance it

    masked_image = run_detection(image, model)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)

    bands = ColourSeparation.getColourBands(masked_image, show_blobs=True)
    ColourSeparation.getResistance(bands)