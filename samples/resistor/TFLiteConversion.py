import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import config as Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from samples.resistor import Resistor

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
RESISTOR_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_resistor_0044.h5") #TODO: Change to model/.h5 directory

config = Resistor.ResistorConfig()

class InferenceConfig(config.__class__):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(RESISTOR_MODEL_PATH, by_name=True)

print("Using TF==", tf.__version__)

keras_model = model.keras_model
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.allow_custom_ops = True
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

converter.optimizations = [ tf.lite.Optimize.DEFAULT ]

tflite_model = converter.convert()

open("model.tflite", "wb").write(tflite_model)