import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
import skimage.io
from time import time
import os
import cv2

#reduce to only 4 colours
n_colors = 4

img_path = "C:\\Users\\Mloong\\Documents\\Code\\OrbitalProject\\Mask_RCNN_TF2_Compatible\\samples\\resistor"

img = []

HSV_boundaries = [
    ([0, 0, 0], [179, 255, 0]), #black, 0
    ([5, 70, 0], [15, 255, 125]), #brown, 1
    ([0, 150, 150], [10, 255, 255]), #red1, 2
    ([165, 150, 150], [179, 255, 255]), #red2, 3
    ([8, 115, 135], [15, 255, 255]), #orange, 4
    ([20, 115, 150], [35, 255, 255]), #yellow, 5
    ([40, 60, 50], [75, 255, 255]), #green, 6
    ([100, 43, 46], [124, 255, 255]), #blue, 7
    ([125, 43, 46], [155, 255, 255]), #violet, 8
    ([0, 0, 46], [179, 43, 220]), #grey, 9
    ([0, 0, 221], [179, 30, 255]), #white, 10
    ([20, 55, 100], [30, 125, 255]), #gold, 11
    ([0, 0, 117], [110, 33, 202]) #silver, 12
]

# Load images
for file in os.listdir(img_path):
    if file.endswith('.jpg') or file.endswith('.png'):
        temp = skimage.io.imread(os.path.join(img_path, file))
        img.append(temp)

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def quantize(img):
    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1])
    img = np.array(img, dtype=np.float64) / 255

    # Load Image and transform to a 2D numpy array.
    w, h, d = original_shape = tuple(img.shape)
    assert d == 3
    image_array = np.reshape(img, (w * h, d))

    print("Fitting model on a small sub-sample of the data")
    t0 = time()
    image_array_sample = shuffle(image_array, random_state=0)[:1000]

    # Using k means algo
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    print("done in %0.3fs." % (time() - t0))

    # Get labels for all points
    print("Predicting color indices on the full image (k-means)")
    t0 = time()
    labels = kmeans.predict(image_array)
    print("done in %0.3fs." % (time() - t0))

    quantized_image = recreate_image(kmeans.cluster_centers_, labels, w, h)

    return quantized_image

Colour_Table = ["black", "brown", "red", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white", "gold", "silver"]

for image in img:
    quantized_image = quantize(image)

    h,w,c = quantized_image.shape

    for i, (lower, upper) in enumerate(HSV_boundaries):
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        mask = cv2.inRange(quantized_image, lower, upper)

        blob = cv2.bitwise_and(image, image, mask=mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for j, contour in enumerate(contours):
            bbox = cv2.boundingRect(contour)

            if (bbox[2]*bbox[3] == h*w):
                # Create a mask for this contour
                contour_mask = np.zeros_like(mask)
                cv2.drawContours(contour_mask, contours, j, 255, -1)

                # Extract the pixels belonging to this contour
                result = cv2.bitwise_and(blob, blob, mask=contour_mask)

                # And draw a bounding box
                top_left, bottom_right = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])
                #print(Boxes)
            
                cv2.rectangle(result, top_left, bottom_right, (255, 255, 255), 2)
                file_name_bbox = f"test-{Colour_Table[i]}-{j}.png"
                cv2.imwrite(file_name_bbox, result)
                print(f" * wrote {file_name_bbox}")