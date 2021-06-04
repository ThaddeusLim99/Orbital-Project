import cv2
import numpy as np

HSV_boundaries = [
    ([0, 0, 0], [179, 255, 0]), #black, 0
    ([5, 70, 0], [15, 255, 125]), #brown, 1
    ([0, 150, 150], [10, 255, 255]), #red1, 2
    #([165, 150, 150], [179, 255, 255]), #red2, 2
    ([8, 115, 135], [15, 255, 255]), #orange, 3
    ([20, 115, 150], [35, 255, 255]), #yellow, 4
    ([40, 60, 50], [75, 255, 255]), #green, 5
    ([100, 43, 46], [124, 255, 255]), #blue, 6
    ([125, 43, 46], [155, 255, 255]), #violet, 7
    ([0, 0, 46], [179, 43, 220]), #grey, 8
    ([0, 0, 221], [179, 30, 255]), #white, 9
    ([20, 55, 100], [30, 125, 255]), #gold, 10
    ([0, 0, 117], [110, 33, 202]) #silver, 11
]

Colour_Table = ["black", "brown", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white", "gold", "silver"]

red2LB =  np.array((165, 150, 150), dtype = "uint8")
red2UB =  np.array((179, 255, 255), dtype = "uint8")
MIN_AREA = 1000

def scale(arr, m):
    for x in arr:
        x *= m
    return arr

def getBackground(image):
    h,w,_ = image.shape
    backgroundColorTop = cv2.mean(image[0:1,:,:])
    backgroundColorBottom = cv2.mean(image[h-1:h,:,:])

    bottomMask = cv2.inRange(image, scale(backgroundColorBottom, 0.6), scale(backgroundColorBottom, 1.4))
    topMask = cv2.inRange(image, scale(backgroundColorTop, 0.6), scale(backgroundColorTop, 1.4))

    backgroundMask = cv2.bitwise_or(topMask, bottomMask)
    backgroundMask = cv2.bitwise_not(backgroundMask)

    return backgroundMask

image = cv2.imread('C:\\Users\\Mloong\\Documents\\Code\\OrbitalProject\\Mask_RCNN_TF2_Compatible\\samples\\resistor\\images\\resistor.jpg')
if image is None:
    print("Image not found")
    exit(-1)

#Resize Image
image = cv2.resize(image, (400,200))

#apply bilateral filter
filtered = cv2.bilateralFilter(image, 5, 80, 80)

#get background mask
background_mask = getBackground(filtered)

#Convert to HSV
image_hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

#Convert to Gray
gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

#edge threshold filters out background and resistor body
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 59, 5)
#ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
thresh = cv2.bitwise_not(thresh)

BoxPos = []

for i, (lower, upper) in enumerate(HSV_boundaries):
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")

    #get mask for each colour
    mask = cv2.inRange(image_hsv, lower, upper)
    #add onto the background mask
    mask = cv2.bitwise_and(background_mask, mask)

    #merge red1 and red2
    if (i == 2):
        mask2 = cv2.inRange(image_hsv, red2LB, red2UB)
        mask = cv2.bitwise_or(mask, mask2, mask)
    
    #add the resistor body mask
    mask = cv2.bitwise_and(mask, thresh, mask=mask)
    #produce masked image
    blob = cv2.bitwise_and(image, image, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for j, contour in enumerate(contours):
        #bbox[0] = x, bbox[1] = y, bbox[2] = w, bbox[3] = h
        bbox = cv2.boundingRect(contour)

        #Filter out contours that are too small
        if (bbox[2]*bbox[3] > MIN_AREA): #and float(bbox[2])/bbox[3] > 0.4):
            # Create a mask for this contour
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, contours, j, 255, -1)

            # Extract the pixels belonging to this contour
            result = cv2.bitwise_and(blob, blob, mask=contour_mask)

            #cv2.imshow(f"{Colour_Table[i]}-{j}", result)
            #cv2.waitKey(0)

            # And draw a bounding box
            top_left, bottom_right = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])
            
            if BoxPos:
                if BoxPos[-1][0][0]-15 <= bbox[0] <= BoxPos[-1][0][0]+15:
                    BoxPos.pop(-1)
            
            # Keep track of all accepted band positions
            BoxPos.append((top_left, Colour_Table[i]))
        
            result = cv2.rectangle(result, top_left, bottom_right, (255, 255, 255), 2)

            #file_name_bbox = f"test-{Colour_Table[i]}-{j}.png"
            #cv2.imwrite(file_name_bbox, result)
            #print(f" * wrote {file_name_bbox}")


print(sorted(BoxPos))