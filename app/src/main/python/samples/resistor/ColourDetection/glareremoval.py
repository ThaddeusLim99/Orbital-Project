#glare removal code from https://rcvaram.medium.com/glare-removal-with-inpainting-opencv-python-95355aa2aa52

import cv2
import numpy as np
from skimage import measure

img_path = r'C:\Users\Thaddeus\Desktop\Mask_RCNN-master\Mask_RCNN-master\samples\resistor\test1-mask0.png'
src = cv2.imread(img_path)

def create_mask(image):
    gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    blurred = cv2.GaussianBlur( gray, (9,9), 0 )
    _,thresh_img = cv2.threshold( blurred, 200, 255, cv2.THRESH_BINARY)
    thresh_img = cv2.erode( thresh_img, None, iterations=2 )
    thresh_img  = cv2.dilate( thresh_img, None, iterations=4 )
    # perform a connected component analysis on the thresholded image,
    # then initialize a mask to store only the "large" components
    #labels = measure.label( thresh_img, neighbors=8, background=0 )
    labels = measure.label( thresh_img, background=0, return_num=False, connectivity=None )
    mask = np.zeros( thresh_img.shape, dtype="uint8" )
    # loop over the unique components
    for label in np.unique( labels ):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros( thresh_img.shape, dtype="uint8" )
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero( labelMask )
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 300:
            mask = cv2.add( mask, labelMask )
    return mask

dst = cv2.inpaint( src, create_mask(src),3,cv2.INPAINT_TELEA) #or INIPAINT_NS

cv2.imshow("mask",create_mask(src))
cv2.imshow("glare removal", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
