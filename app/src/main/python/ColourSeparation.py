import base64
import cv2
import numpy as np
import os
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

sys.path.append(ROOT_DIR)
#import ResistanceCalculator as res
#from ColourDetection import ColourQuantizationKmeans

colourCode = {
    "black": 0,
    "brown": 1,
    "red": 2,
    "orange": 3,
    "yellow": 4,
    "green": 5,
    "blue": 6,
    "violet": 7,
    "grey": 8,
    "white": 9,
    "gold": -1,
    "silver": -2
}

toleranceCode = {
    "brown": 0.01,
    "red": 0.02,
    "orange": 0.0005,
    "yellow": 0.0002,
    "green": 0.005,
    "blue": 0.0025,
    "violet": 0.001,
    "grey": 0.0001,
    "gold": 0.05,
    "silver": 0.1
}

ppmCode = {
    "black": 250,
    "brown": 100,
    "red": 50,
    "orange": 15,
    "yellow": 25,
    "green": 20,
    "blue": 10,
    "violet": 5,
    "grey": 1
}

def threeBandCalc(first, second, m):
    res = (colourCode[first] * 10 + colourCode[second]) * 10**colourCode[m]
    lb = res * 0.8
    ub = res * 1.2
    return res, lb, ub

def fourBandCalc(first, second, m, tol):
    res = (colourCode[first] * 10 + colourCode[second]) * 10**colourCode[m]
    lb = res * (1 - toleranceCode[tol])
    ub = res * (1 + toleranceCode[tol])
    return res, lb, ub

def fiveBandCalc(first, second, third, m, tol):
    res = (colourCode[first] * 100 + colourCode[second] * 10 + colourCode[third]) * 10**colourCode[m]
    lb = res * (1 - toleranceCode[tol])
    ub = res * (1 + toleranceCode[tol])
    return res, lb, ub

def sixBandCalc(first, second, third, m, tol, ppm):
    res = (colourCode[first] * 100 + colourCode[second] * 10 + colourCode[third]) * 10**colourCode[m]
    lb = res * (1 - toleranceCode[tol])
    ub = res * (1 + toleranceCode[tol])
    return res, lb, ub, ppmCode[ppm]

HSV_boundaries = [
    ([0, 0, 20], [179, 255, 35]), #black, 0
    ([5, 70, 60], [15, 255, 125]), #brown, 1
    ([0, 150, 150], [10, 255, 255]), #red1, 2
    #([165, 150, 150], [179, 255, 255]), #red2, 2
    ([8, 115, 135], [15, 255, 255]), #orange, 3
    ([20, 130, 175], [35, 255, 255]), #yellow, 4
    ([40, 60, 50], [75, 255, 255]), #green, z
    ([100, 43, 46], [124, 255, 255]), #blue, 6
    ([125, 43, 46], [155, 255, 255]), #violet, 7
    ([0, 0, 80], [179, 40, 200]), #grey, 8
    ([0, 0, 221], [179, 30, 255]), #white, 9
    ([20, 55, 100], [30, 125, 255]), #gold, 10
    ([0, 0, 117], [110, 33, 202]) #silver, 11
]

Colour_Table = ["black", "brown", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white", "gold", "silver"]

red2LB =  np.array((165, 150, 150), dtype = "uint8")
red2UB =  np.array((179, 255, 255), dtype = "uint8")
#TODO: Change values until results are accurate
MIN_AREA = 1000

def scale(arr, m):
    for x in arr:
        x *= m
    return arr

# Get the average background colour by using the top row and bottom row
# Works well if the background is a solid colour and consistent
def getBackground(image):
    h,w,_ = image.shape
    meanTopBackgroundColor = cv2.mean(image[0:1,:,:])
    meanBotBackgroundColor = cv2.mean(image[h-1:h,:,:])

    botMask = cv2.inRange(image, scale(meanBotBackgroundColor, 0.5), scale(meanBotBackgroundColor, 1.5))
    topMask = cv2.inRange(image, scale(meanTopBackgroundColor, 0.5), scale(meanTopBackgroundColor, 1.5))

    backgroundMask = cv2.bitwise_or(topMask, botMask)
    backgroundMask = cv2.bitwise_not(backgroundMask)

    return backgroundMask

# Quantizes the image using openCV's k means algorithm
def quantize(img):
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Number of colours to reduce to
    K = 32

    # k means algorithm
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res_reshaped = res.reshape((img.shape))

    return res_reshaped

# Function taken from https://github.com/lucasmoraeseng/resistordetector
def GetDrawResistor(DataIn):
    Data = DataIn.copy()
    t = np.uint8(np.mean(Data,axis=0))

    DrawResistor = np.zeros((Data.shape[0],Data.shape[1],3), np.uint8)
    DRWidth = Data.shape[1]
    DRHeight = Data.shape[0]

    for i in range((np.shape(t)[0])):
        cv2.rectangle(Data,(i,0), (i+1,DRHeight), np.float64(t[i]), 1)

    return Data

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    # if both are filled, take the height still
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

# Returns the colour bands from left to right in the image, order is not necessarily correct
def getColourBands(image, show_blobs=False, save_blobs=False):
    #Resize Image
    #image = cv2.resize(image, (400,200))
    image = image_resize(image, width=400)

    #quantize the image
    image = quantize(image)


    #image = GetDrawResistor(image)

    #apply bilateral filter
    filtered = cv2.bilateralFilter(image, 5, 80, 80)

    #get background mask
    background_mask = getBackground(filtered)

    #Convert to HSV
    image_hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

    #Convert to Gray
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    #edge threshold filters out background and resistor body
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 71, 5)#59, 5)
    thresh = cv2.bitwise_not(thresh)


    #yellow usually gets filtered out after the thresholding
    #add a tighter yellow hsv boundary to reduce adding in resistor body
    yellowLB = np.array((20, 150, 190), dtype = "uint8")
    yellowUB = np.array((35, 255, 255), dtype = "uint8")
    yellowMask = cv2.inRange(image_hsv, yellowLB, yellowUB)
    # new threshold mask
    thresh = cv2.bitwise_or(thresh, yellowMask)

    if show_blobs:
        test = cv2.bitwise_and(image, image, mask=thresh)
        cv2.imshow("test", test)
        cv2.waitKey(0)

    if save_blobs:
        test = cv2.bitwise_and(image, image, mask=thresh)
        cv2.imwrite("test.png", test)

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

            #Exclude contours that are too small
            if (bbox[2]*bbox[3] > MIN_AREA): #and float(bbox[2])/bbox[3] > 0.4):
                # Create a mask for this contour
                contour_mask = np.zeros_like(mask)
                cv2.drawContours(contour_mask, contours, j, 255, -1)

                # Extract the pixels belonging to this contour
                result = cv2.bitwise_and(blob, blob, mask=contour_mask)

                # And draw a bounding box
                top_left, bottom_right = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])

                # Check if the the bounding box is referring to the same colour band
                if BoxPos:
                    for (pos), colour in BoxPos:
                        if colour == Colour_Table[i]:
                            if pos[0]-20 <= bbox[0] <= pos[0]+20:
                                BoxPos.remove((pos, Colour_Table[i]))
                    #if BoxPos[-1][0][0]-20 <= bbox[0] <= BoxPos[-1][0][0]+20:
                    #    BoxPos.pop(-1)

                # Keep track of all accepted band positions
                BoxPos.append((top_left, Colour_Table[i]))

                result = cv2.rectangle(result, top_left, bottom_right, (255, 255, 255), 2)

                if show_blobs:
                    cv2.imshow(f"{Colour_Table[i]}-{j}", result)
                    cv2.waitKey(0)

                if save_blobs:
                    file_name_bbox = f"test-{Colour_Table[i]}-{j}.png"
                    cv2.imwrite(file_name_bbox, result)
                    print(f" * wrote {file_name_bbox}")

    return BoxPos

# Prints the resistance from the given input colour bands
def getResistance(BoxPos):
    numOfBands = len(BoxPos)

    # Invalid Readings
    if numOfBands < 3:
        print("Not enough colour bands detected")
        return
    if numOfBands > 6:
        print("Too many colours detected")
        return

    # Gold/Silver bands can only describe tolerance levels
    if numOfBands == 3:
        BoxPos = sorted(BoxPos)

    elif 3 < numOfBands < 6:
        if min(BoxPos)[1] == "gold" or min(BoxPos)[1] == "silver":
            BoxPos = sorted(BoxPos, reverse=True)
        else:
            BoxPos = sorted(BoxPos)

    elif numOfBands == 6:
        if BoxPos[2][1] == "gold" or BoxPos[2][1] == "silver":
            BoxPos = sorted(BoxPos, reverse=True)
        else:
            BoxPos = sorted(BoxPos)

    print("Sorted order of all of the colour bands detected")

    print("Number of colour bands detected:", numOfBands)
    print("Position and Colours:", BoxPos)

    #Get Resistance
    if numOfBands == 3:
        results = threeBandCalc(BoxPos[0][1], BoxPos[1][1], BoxPos[2][1])
    elif numOfBands == 4:
        results = fourBandCalc(BoxPos[0][1], BoxPos[1][1], BoxPos[2][1], BoxPos[3][1])
    elif numOfBands == 5:
        results = fiveBandCalc(BoxPos[0][1], BoxPos[1][1], BoxPos[2][1], BoxPos[3][1], BoxPos[4][1])
    elif numOfBands == 6:
        results = sixBandCalc(BoxPos[0][1], BoxPos[1][1], BoxPos[2][1], BoxPos[3][1], BoxPos[4][1], BoxPos[5][1])

    temp = list(results)

    for i in range(len(results)):
        if temp[i] >= 1000000:
            temp[i] /= 1000000
            temp[i] = str(temp[i]) + " MOhms"
        elif temp[i] >= 1000:
            temp[i] /= 1000
            temp[i] = str(temp[i]) + " kOhms"
        else:
            temp[i] = str(temp[i]) + " Ohms"

    result = tuple(temp)

    if numOfBands <= 5:
        print("Resistance:", result[0], "LB:", result[1], "UB:", result[2])
    elif numOfBands == 6:
        print("Resistance:", result[0], "LB:", result[1], "UB:", result[2], "PPM:", results[3])

    #returns resistor value and the colours detected respectively
    return result[0], BoxPos

def main(mask):
    #decode string that was passed
    decoded_data = base64.b64decode(mask)
    #convert decoded data to numpy data
    np_data = np.fromstring(decoded_data,np.uint8)
    #convert to cv2 image
    image = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)

    if image is None:
        print("Image not found")
        exit(-1)

    bands = getColourBands(image, save_blobs=True)
    return getResistance(bands)