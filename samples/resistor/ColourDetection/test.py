import cv2
import numpy as np

# Minimum percentage of pixels of same hue to consider dominant colour
MIN_PIXEL_CNT_PCT = (1.0/20.0)

image = cv2.imread('images\\11-mask1.jpg')
if image is None:
    print("Failed to load iamge.")
    exit(-1)

#Convert to HSV
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("img", image_hsv)
cv2.waitKey(0)

# We're only interested in the hue
h,s,v = cv2.split(image_hsv)
print(h,s,v)

HSV_codes = {
    "black_lb":[0, 0, 0],
    "black_ub":[255, 255, 0],
    "brown_lb":[10, 100, 20],
    "brown_ub":[20, 255, 200],
    "ORANGE_LB": np.array([7, 50, 50],np.uint8),
    "ORANGE_UB": np.array([12, 255, 255],np.uint8)
}

# Let's count the number of occurrences of each hue [0,179]
bins = np.bincount(h.flatten())
print(bins)

# And then find the dominant hues
# Peaks are the indices where the condition is fulfilled
peaks = np.where(bins > (h.size * MIN_PIXEL_CNT_PCT))[0]
print(peaks)

# Now let's find the shape matching each dominant hue
for i, peak in enumerate(peaks):
    # First we create a mask selecting all the pixels of this hue
    mask = cv2.inRange(h, int(peak), int(peak))

    # And use it to extract the corresponding part of the original colour image
    blob = cv2.bitwise_and(image, image, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for j, contour in enumerate(contours):
        bbox = cv2.boundingRect(contour)

        # Create a mask for this contour
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, contours, j, 255, -1)
        print(f"Found hue {peak} in region {bbox}.")

        # Extract and save the area of the contour
        region = blob.copy()[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        region_mask = contour_mask[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        region_masked = cv2.bitwise_and(region, region, mask=region_mask)
        file_name_section = "colourblobs-%d-hue_%03d-region_%d-section.png" % (i, peak, j)
        #cv2.imwrite(file_name_section, region_masked)
        print (f"* wrote {file_name_section}")

        # Extract the pixels belonging to this contour
        result = cv2.bitwise_and(blob, blob, mask=contour_mask)

        # And draw a bounding box
        top_left, bottom_right = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])
        if (bbox[2]*bbox[3] > 200):
            cv2.rectangle(result, top_left, bottom_right, (255, 255, 255), 2)
            file_name_bbox = "colourblobs-%d-hue_%03d-region_%d-bbox.png" % (i, peak, j)
            cv2.imwrite(file_name_bbox, result)
            print(f" * wrote {file_name_bbox}")