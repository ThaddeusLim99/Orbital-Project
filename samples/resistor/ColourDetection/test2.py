import cv2
import numpy as np

# Minimum percentage of pixels of same hue to consider dominant colour
MIN_PIXEL_CNT_PCT = (1.0/20.0)

image = cv2.imread('images\\test1-mask0.png')
if image is None:
    print("Failed to load iamge.")
    exit(-1)

#Convert to HSV
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("img", image_hsv)
cv2.waitKey(0)
cv2.imwrite("hsv_image.png", image_hsv)

# We're only interested in the hue
h,s,v = cv2.split(image_hsv)
print(h,s,v)

HSV_codes = {
    "BLACK_LB":np.array([0, 0, 0], np.uint8),
    "BLACK_UB":np.array([255, 255, 0], np.uint8),
    "BROWN_LB":np.array([10, 100, 20], np.uint8),
    "BROWN_UB":np.array([20, 255, 200], np.uint8),
    "ORANGE_LB": np.array([8, 50, 50],np.uint8),
    "ORANGE_UB": np.array([15, 255, 255],np.uint8),
    "YELLOW_LB": np.array([20, 100, 100], np.uint8),
    "YELLOW_UB": np.array([30, 255, 255], np.uint8)
}

# Let's count the number of occurrences of each hue [0,179]
bins = np.bincount(h.flatten())
print(bins)

# And then find the dominant hues
# Peaks are the indices where the condition is fulfilled
peaks = np.where(bins > (h.size * MIN_PIXEL_CNT_PCT))[0]
print(peaks)

mask = cv2.inRange(image_hsv, HSV_codes["YELLOW_LB"], HSV_codes["YELLOW_UB"])

blob = cv2.bitwise_and(image, image, mask=mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
for j, contour in enumerate(contours):
    bbox = cv2.boundingRect(contour)

    if (bbox[2]*bbox[3] > 1000):
        # Create a mask for this contour
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, contours, j, 255, -1)

        # Extract and save the area of the contour
        region = blob.copy()[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        region_mask = contour_mask[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        region_masked = cv2.bitwise_and(region, region, mask=region_mask)

        # Extract the pixels belonging to this contour
        result = cv2.bitwise_and(blob, blob, mask=contour_mask)

        # And draw a bounding box
        top_left, bottom_right = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])
    
        cv2.rectangle(result, top_left, bottom_right, (255, 255, 255), 2)
        file_name_bbox = f"test{j}.png"
        cv2.imwrite(file_name_bbox, result)
        print(f" * wrote {file_name_bbox}")    