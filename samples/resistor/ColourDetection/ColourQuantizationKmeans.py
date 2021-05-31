import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
import skimage.io
from time import time

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help = "path to the image")
ap.add_argument("-n", "--ncolors", required=False, default=64, help = "Number of colours to quantize to")
args = vars(ap.parse_args())

n_colors = int(args["ncolors"])

# Load the Image
img = skimage.io.imread(args["image"])

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

# Using some k means algo
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))

'''
# Using some random quantizing algo
codebook_random = shuffle(image_array, random_state=0)[:n_colors]
print("Predicting color indices on the full image (random)")
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random,
                                          image_array,
                                          axis=0)
print("done in %0.3fs." % (time() - t0))
'''

# Display all results, alongside original image
#plt.figure(1)
#plt.clf()
#plt.axis('off')
#plt.title('Original image (96,615 colors)')
#plt.imshow(img)

#plt.figure(2)
#plt.clf()
plt.axis('off')
#title = f"Quantized image ({n_colors}, K-Means)"
#plt.title(title)
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
plt.savefig("quantized_image.png", bbox_inches="tight", pad_inches=-0.2,  orientation='landscape')
#plt.show()