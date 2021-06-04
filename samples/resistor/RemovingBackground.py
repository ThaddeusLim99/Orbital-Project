#Only works well if the background is a solid colour

import cv2
import numpy as np

def scale(arr, m):
    for x in arr:
        x *= m
    return arr

def getBackground(image):
    h,w,_ = image.shape
    backgroundColorTop = cv2.mean(image[0:10,:,:])
    backgroundColorBottom = cv2.mean(image[h-10:h,:,:])

    print(backgroundColorBottom)
    print(backgroundColorTop)

    bottomMask = cv2.inRange(image, scale(backgroundColorBottom, 0.6), scale(backgroundColorBottom, 1.4))
    topMask = cv2.inRange(image, scale(backgroundColorTop, 0.6), scale(backgroundColorTop, 1.4))

    backgroundMask = cv2.bitwise_or(topMask, bottomMask)

    return backgroundMask

def getReflections(image):
    lower = np.array((0, 0, 200), dtype = "uint8")
    upper = np.array((180, 256, 256), dtype = "uint8")
    mask = cv2.inRange(image, lower, upper)
    mask = cv2.bitwise_not(mask)
    mask = cv2.erode(mask, None, anchor=(-1,-1), iterations=2)
    mask = cv2.bitwise_not(mask)

    return mask

def getResistor(image):
    reflectionMask = getReflections(image)
    backgroundMask = getBackground(image)
    resistorMask = cv2.bitwise_or(reflectionMask, backgroundMask)
    resistorMask = cv2.bitwise_not(resistorMask)

    return resistorMask

def getResistance(image):
    h,w,_ = image.shape

    image = cv2.bilateralFilter(image, 5, 80, 80)
    cv2.cvtColor(image, image, cv2.COLOR_BGR2HSV)

    resistorMask = getResistor(image)
    
    

image = cv2.imread("C:\\Users\\Mloong\\Documents\\Code\\OrbitalProject\\Mask_RCNN_TF2_Compatible\\datasets\\resistor\\val\\LEDs.png")
image = cv2.resize(image, (500, 500))
kernal = np.ones((5, 5), "uint8") 

mask = getResistor(image)
#mask = cv2.bitwise_not(mask)
blob = cv2.bitwise_and(image, image, mask=mask)

#cv2.imwrite("img.png", blob)
cv2.imshow("img", blob)
cv2.waitKey(0)