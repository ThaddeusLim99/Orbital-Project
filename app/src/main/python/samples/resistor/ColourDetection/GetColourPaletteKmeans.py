from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

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
    ([20, 55, 100], [45, 125, 255]), #gold, 11
    ([0, 0, 117], [110, 33, 202]) #silver, 12
]

Colour_Table = ["black", "brown", "red", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white", "gold", "silver"]

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
    #print(ordered_colors)
    #print(rgb_colors)

    #get colour palette + most dominant colour in the image
    colors_list = [0 for x in range(0, 13)]
    for i, colors in enumerate(rgb_colors):
        print(f"=========RGB values for {i}=========")
        print(colors[0], colors[1], colors[2])
        print(f"=========HSV values for {i}=========")
        print(rgb2hsv(colors[0], colors[1], colors[2]))
        h,s,v = rgb2hsv(colors[0], colors[1], colors[2])
        for j, (lower, upper) in enumerate(HSV_boundaries):
            if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
                colors_list[j] += 1
    print("========================Dominant Colour========================")
    dominant_color = colors_list.index(max(colors_list))
    print(dominant_color)
    print(colors_list)
    print(Colour_Table[dominant_color])

    if (show_chart):
        plt.figure(num=1, figsize = (20, 20))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
        plt.savefig(f'colourDetection-{i}.png', bbox_inches='tight', pad_inches=-0.5)
        #plt.show()

    return rgb_colors

def rgb2hsv(r,g,b):
    r_p = r/255
    g_p = g/255
    b_p = b/255
    
    Cmax = max(r_p,g_p,b_p)
    Cmin = min(r_p,g_p,b_p)
    delta = Cmax - Cmin
    # HSV in terms of openCV
    if Cmax == r_p:
        hue = 30 * (((g_p - b_p)/delta) % 6)
    elif Cmax == g_p:
        hue = 30 * (((b_p - r_p)/delta) + 2)
    elif Cmax == b_p:
        hue = 30 * (((r_p - g_p)/delta) + 4)
    else:
        hue = 0

    if Cmax == 0:
        sat = 0
    else:
        sat = delta/Cmax * 255

    value = Cmax * 255

    if hue > 179:
        hue = 179

    return hue,sat,value

IMAGE_DIRECTORY = "C:\\Users\\Mloong\\Documents\\Code\\OrbitalProject\\Mask_RCNN_TF2_Compatible\\samples\\resistor"
images = []

#iterate through all files in the directory and add all files ending with .jpg
for file in os.listdir(IMAGE_DIRECTORY):
    if file.endswith('.jpg') or file.endswith('.png'):
        images.append(get_image(os.path.join(IMAGE_DIRECTORY, file)))

#iterate through every image and create colour piechart
for i, image in enumerate(images):
    get_colours(image, 5, show_chart=True)
    #cv2.imshow("image", image)
    #cv2.waitKey(0)
