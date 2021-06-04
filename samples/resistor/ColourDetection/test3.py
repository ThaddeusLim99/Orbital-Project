import cv2
import numpy as np

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

locationValues = {}

def findLocations(searchMat):
    global locationValues
    locationValues = {}
    areas = {}

    for i, (lower, upper) in enumerate(HSV_boundaries):
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        mask = cv2.inRange(searchMat, lower, upper)

        contours, heirarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 20:
                M = cv2.moments(contour)
                cx = M['m10']/M['m00']
                shouldStoreLocation = True
                for j in range(len(locationValues)):
                    keys = list(locationValues.keys())
                    #print(keys, cx)
                    if (abs(float(keys[j]) - cx) < 10):
                        print(areas[f"{keys[j]}"], area)
                        if areas[f"{keys[j]}"] > area:
                            shouldStoreLocation = False
                            break
                        else:
                            locationValues.pop(f"{keys[j]}")
                            areas.pop(f"{keys[j]}")
                if shouldStoreLocation:
                    areas[f"{cx}"] = area
                    locationValues[f"{cx}"] = area

def processFrame(img_path):
    image = cv2.imread(img_path)
    h,w,_ = image.shape

    roi = image[h//2:h//2+30, 50:w-50]

    filtered = roi.copy()

    filtered = cv2.bilateralFilter(roi, 5, 80, 80)

    filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

    findLocations(filtered)

    if (len(locationValues) >= 3):
        keys = list(locationValues.keys())
        k_tens = keys[0]
        k_units = keys[1]
        k_power = keys[2]

        value = 10*locationValues[f"{k_tens}"] + locationValues[f"{k_units}"]
        print(locationValues[f"{k_power}"])
        value *= (10 ** locationValues[f"{k_power}"])

        print(value)

img_path = "C:\\Users\\Mloong\\Documents\\Code\\OrbitalProject\\Mask_RCNN_TF2_Compatible\\samples\\resistor\\images\\resistor.jpg"
processFrame(img_path)