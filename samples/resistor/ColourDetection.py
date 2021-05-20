from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

#converts RGB to HEX format
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

#returns the image after converting it to RGB
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("The type of this input is {}".format(type(image)))
    print("Shape: {}".format(image.shape))
    return image

#breakdowns all the colours available in the image and represent them on a pie chart
def get_colours(image, number_of_colors, show_chart=True):
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)

    return rgb_colors


IMAGE_DIRECTORY = os.path.abspath(".")
img_dir = os.path.join(IMAGE_DIRECTORY, "images")
images = []

#iterate through all files in the directory and add all files ending with .jpg
for file in os.listdir(img_dir):
    if file.endswith('.jpg'):
        images.append(get_image(os.path.join(img_dir, file)))

plt.figure(figsize=(20, 10))
#iterate through every image and get colour piechart
for image in images:
    get_colours(image, 10, show_chart=True)
    cv2.imshow("image", image)
    plt.show()
    cv2.waitKey(0)