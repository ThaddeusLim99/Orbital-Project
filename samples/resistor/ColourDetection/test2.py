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

# Load images
for file in os.listdir(img_path):
    if file.endswith('.jpg') or file.endswith('.png'):
        temp = skimage.io.imread(os.path.join(img_path, file))
        img.append(temp)

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

Colour_Table = ["black", "brown", "red", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white", "gold", "silver"]

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d), dtype=np.float32)
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]] #* 255
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

    image_array_sample = shuffle(image_array, random_state=0)[:1000]

    # Using k means algo
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

    # Get labels for all points
    labels = kmeans.predict(image_array)

    #save image
    result = recreate_image(kmeans.cluster_centers_, labels, w, h)
    plt.axis('off')
    #cv2.imshow("img", result)
    #cv2.waitKey(0)

    plt.imshow(result)
    plt.savefig("./images/quantized_image.png", bbox_inches="tight", pad_inches=-0.2,  orientation='landscape')
    #plt.show()
    '''
    for x in range(result.shape[0]):
        result[x, :, :] = np.fliplr(result[x, :, :])

    return result
    '''

for k, image in enumerate(img):
    #image = quantize(image)
    #print(image.shape)
    
    quantize(image)
    image = cv2.imread('C:\\Users\\Mloong\\Documents\\Code\\OrbitalProject\\Mask_RCNN_TF2_Compatible\\samples\\resistor\\images\\quantized_image.png')
    #print(image.shape)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i, (lower, upper) in enumerate(HSV_boundaries):
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        mask = cv2.inRange(image_hsv, lower, upper)

        blob = cv2.bitwise_and(image, image, mask=mask)
        
        h,w = mask.shape

        if 0.5*h*w < np.count_nonzero(mask):
            #cv2.imshow(f"{Colour_Table[i]}", blob)
            #cv2.waitKey(0)
            file_name_bbox = f"image-{k}-{Colour_Table[i]}.png"
            #cv2.imwrite(f"{file_name_bbox}", blob)
            print(f" * colour for image {k} is {Colour_Table[i]}")

        '''
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for j, contour in enumerate(contours):
            bbox = cv2.boundingRect(contour)

            #print(h*w)
            #print(bbox[2]*bbox[3])

            if (0.4*h*w <= bbox[2]*bbox[3] <= h*w):
                file_name_bbox = f"image-{k}-{Colour_Table[i]}-{j}.png"
                cv2.imwrite(f"{file_name_bbox}", blob)
                print(f" * colour for image {k} is {Colour_Table[i]}")
        '''