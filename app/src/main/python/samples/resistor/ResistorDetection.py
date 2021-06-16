#ResistorDetectiontftwo
# Runs a trained model, evaluating an image from val or a unique image then saving it as a .jpg
import base64
import io
from PIL import Image
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

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
#renamed DEFAULT_LOGS_DIR to MODEL_DIR
from os.path import dirname,join
MODEL_DIR = join(dirname(__file__), "../../logs")

#added a direct way to find h5 file
# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_resistor_0044.h5")
#from os.path import dirname,join
COCO_MODEL_PATH = join(dirname(__file__),"../../mask_rcnn_resistor_0044.h5")

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

def run_detection(model,image):
    print("image_path = image")
    image_path = image
    print("cv2.cvtColor")
    # Run object detection
    #img = skimage.io.imread(image_path)
    image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    print("model.detect")
    try:
        print("trying model.detect")
        results = model.detect([image])
    except:
        print("model.detect error")
        return
    if results is None:
        print("No detection")
        return
    else:
        print("detected!")
    # Display results
    ax = get_ax(1)
    try:
        print("trying results[0]")
        r = results[0]
    except:
        print("results[0] error")
        return
    # print(r)
    #print the coordinates of the first bounding box, format (y1, x1, y2, x2)
    #print (r['rois'][0])

    if r['masks'].size > 0:
        print("loading masks")
        masks = np.zeros((image.shape[0], image.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
        for m in range(r['masks'].shape[-1]):
            masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 
                                        (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        y_scale = image.shape[0]/1024
        x_scale = image.shape[1]/1024
        rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)
        print("finished with masks")
        masks, rois = refine_masks(masks, rois)
    else:
        print("finished with masks too")
        masks, rois = r['masks'], r['rois']

    #visualize.display_instances(image, rois, masks, r['class_ids'],
    #                            ['bg'] + ['resistor'], r['scores'], #since we are only detecting resistors
    #                            ax=ax, title="Predictions")

    #Save the mask overlaid on the image
    #name = args.name
    #plt.savefig(f"{name}.png",bbox_inches='tight', pad_inches=-0.5, orientation='landscape')
    #cv2.imwrite(f"{name}.png", image)

    mask = r['masks']
    mask = mask.astype(int)
    print("masking")
    #print("==========Mask Shape==========")
    #print(mask.shape)

    #print("==========Image Shape==========")
    #print(image.shape)

    #Extract the masks
    for i in range(mask.shape[2]):
        temp = image.copy()
        print("image.copy")
        for j in range(temp.shape[2]):
            temp[:,:,j] = temp[:,:,j] * mask[:,:,i]
            print("temp ??")
        #Crop the image to only fit the bounding box
        #if args.crop:
            #print("==========Coordinates of Bounding Box in the form of [y1 x1 y2 x2]==========")
            #print(r['rois'][i])
        y1, x1, y2, x2 = r['rois'][i]
        print(y1)
        print(x1)
        print(y2)
        print(x2)
        temp = temp[y1:y2, x1:x2]

        plt.figure(figsize=(8,8))
        #Save the masked image for each mask
        #plt.imshow(temp)
        #plt.savefig(f'{name}-mask{i}.jpg',bbox_inches='tight', pad_inches=-0.5,orientation= 'landscape')
        #cv2.imwrite(f'{name}-mask{i}.png', temp)
        #print(temp)
        #convert to image to PIL image
        pil_im = Image.fromarray(temp)
        #print(pil_im)
        #convert image to bytes
        buff = io.BytesIO()
        pil_im.save(buff,format="PNG")
        #convert again to base64 string
        img_str = base64.b64encode(buff.getvalue())
        print("LOLOL")
        if img_str is None:
            print("img_str is null")
        print("img_str is not null")
        #print(str(img_str,'utf-8'))
        return ""+str(img_str,'utf-8')

    plt.close()

#if __name__ == '__main__':
#    import argparse

    # Parse command line arguments
    #parser = argparse.ArgumentParser(
        #description='Extracting the masks of an Image, one for the mask overlaid over the image, and one for each individual masked image')
    #parser.add_argument('--logs', required=False,
     #                   default=MODEL_DIR, #changed DEFAULT_LOGS_DIR to MODEL_DIR
      #                  metavar="/path/to/logs/",
       #                 help='Logs and checkpoints directory (default=logs/)')
    #parser.add_argument('--device', required=False,
     #                   default="/cpu:0",
      #                  help="Device to load the neural network on. Useful if you're training a model on the same machine, in which case use CPU and leave the GPU for training. (ie. /cpu:0 or /gpu:0)")
    #parser.add_argument('--weights', required=True,
     #                   metavar="/path/to/weights.h5",
      #                  help="Path to weights .h5 file")
    #parser.add_argument('--crop', required=False,
     #                   default=True,
      #                  metavar="/path/to/logs/",
       #                 help='Crop the image such that only contents of the bounding box is shown')
    #parser.add_argument('--image', required=True,
     #                   metavar="path or URL to image",
      #                  help='Image to run the detection on')
    #parser.add_argument('--name', required=False,
     #                   default="image",
      #                  metavar="name of the saved images",
       #                 help='Image to apply the color splash effect on')
    #args = parser.parse_args()
def main(data):
    print("owadio!")
    #print(data)
    #decode string that was passed
    decoded_data = base64.b64decode(data)
    #convert decoded data to numpy data
    np_data = np.fromstring(decoded_data,np.uint8)
    #convert to cv2 image
    image = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)

    config = Resistor.ResistorConfig()

    class InferenceConfig(config.__class__):
    # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    
    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    #print("Logs: ", args.logs)

    #changed this part
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    #WEIGHTS_PATH = args.weights
    # Download COCO trained weights from Releases if needed
    #if not os.path.exists(WEIGHTS_PATH):
    #    print("h5 file not found")

    # Load weights trained on MS-COCO
    #print("Loading weights ", args.weights)
    #changed this part
    try:
        print("trying model.load_weights")
        model.load_weights(COCO_MODEL_PATH, by_name=True)
    except:
        print("mdel.load_weights error")
        return
    print("before run_detection")
    return run_detection(model,image)