import cv2
import numpy as np
import os
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from samples.resistor import Resistor
from mrcnn import utils
import mrcnn.model as modellib

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

RESISTOR_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_resistor_0044.h5")

if not os.path.exists(RESISTOR_MODEL_PATH):
    utils.download_trained_weights(RESISTOR_MODEL_PATH)

class InferenceConfig(Resistor.ResistorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(RESISTOR_MODEL_PATH, by_name=True)
class_names = [
    'BG', 'resistor'
]


def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

colors = random_colors(len(class_names))
class_dict = {
    name: color for name, color in zip(class_names, colors)
}

def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image


if __name__ == '__main__':
    """
        test everything
    """
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Cannot open camera")
        exit()

    #input_video = sys.argv[1]
    #capture = cv2.VideoCapture(input_video)

    # these 2 lines can be removed if you dont have a 1080p camera.
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Recording Video
    fps = 25.0
    width = int(capture.get(3))
    height = int(capture.get(4))
    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter("recording_video.avi", fcc, fps, (width, height))

    while True:
        ret, frame = capture.read()

        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        cv2.imshow('frame', frame)

        # Recording Video
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()