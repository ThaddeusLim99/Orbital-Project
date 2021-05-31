import cv2
import numpy as np

image = cv2.imread('C:\\Users\\Mloong\\Documents\\Code\\OrbitalProject\\Mask_RCNN_TF2_Compatible\\samples\\resistor\\quantized_image.png')
if image is None:
    print("Failed to load iamge.")
    exit(-1)

#Convert to HSV
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

'''
HSV_codes = {
    "BLACK_LB":np.array([0, 0, 0], np.uint8),
    "BLACK_UB":np.array([179, 255, 0], np.uint8),
    "BROWN_LB":np.array([10, 100, 20], np.uint8),
    "BROWN_UB":np.array([20, 255, 200], np.uint8),
    "RED1_LB": np.array([0, 43, 46],np.uint8),
    "RED1_UB": np.array([10, 255, 255],np.uint8),
    "RED2_LB": np.array([156, 43, 46],np.uint8),
    "RED2_UB": np.array([179, 255, 255],np.uint8),
    "ORANGE_LB": np.array([8, 50, 50],np.uint8),
    "ORANGE_UB": np.array([15, 255, 255],np.uint8),
    "YELLOW_LB": np.array([20, 100, 100], np.uint8),
    "YELLOW_UB": np.array([30, 255, 255], np.uint8),
    "GREEN_LB": np.array([45, 100, 50],np.uint8),
    "GREEN_UB": np.array([75, 255, 255],np.uint8),
    "BLUE_LB": np.array([100, 43, 46],np.uint8),
    "BLUE_UB": np.array([124, 255, 255],np.uint8),
    "VIOLET_LB": np.array([125, 43, 46],np.uint8),
    "VIOLET_UB": np.array([155, 255, 255],np.uint8),
    "GREY_LB": np.array([0, 0, 46],np.uint8),
    "GREY_UB": np.array([179, 43, 220],np.uint8),
    "WHITE_LB": np.array([0, 0, 221],np.uint8),
    "WHITE_UB": np.array([179, 30, 255],np.uint8),
    "GOLD_LB": np.array([20, 181, 193],np.uint8),
    "GOLD_UB": np.array([25, 255, 255],np.uint8),
    "SILVER_LB": np.array([0, 0, 117],np.uint8),
    "SILVER_UB": np.array([110, 33, 202],np.uint8)
}
'''

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

Boxes = []

h,w,c = image.shape

Colour_Table = ["black", "brown", "red", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white", "gold", "silver"]

kernal = np.ones((5, 5), "uint8") 

for i, (lower, upper) in enumerate(HSV_boundaries):
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")

    mask = cv2.inRange(image_hsv, lower, upper)
    mask = cv2.erode(mask, kernal)
    #mask = cv2.inRange(image_hsv, HSV_codes["VIOLET_LB"], HSV_codes["VIOLET_UB"])

    blob = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow(f"{Colour_Table[i]}", blob)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for j, contour in enumerate(contours):
        bbox = cv2.boundingRect(contour)

        if (bbox[2]*bbox[3] > 1000 and bbox[2]*bbox[3] < h*w):
            # Create a mask for this contour
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, contours, j, 255, -1)

            # Extract the pixels belonging to this contour
            result = cv2.bitwise_and(blob, blob, mask=contour_mask)

            # And draw a bounding box
            top_left, bottom_right = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])
            Boxes.append((Colour_Table[i] ,top_left, bottom_right))
            #print(Boxes)
        
            cv2.rectangle(result, top_left, bottom_right, (255, 255, 255), 2)
            file_name_bbox = f"test-{Colour_Table[i]}-{j}.png"
            cv2.imwrite(file_name_bbox, result)
            print(f" * wrote {file_name_bbox}")

print(Boxes)