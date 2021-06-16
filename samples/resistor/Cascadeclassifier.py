import numpy as np
import cv2
import PIL
import sklearn
import random
from matplotlib import pyplot as plt
import sys
import os

ROOT_DIR = os.path.abspath(".")

sys.path.append(ROOT_DIR)
from ColourDetection import ColourSeparation

objCascade = cv2.CascadeClassifier("ResNew22_cascade.xml")

### Taken from https://github.com/lucasmoraeseng/resistordetector
def FindInImage(Sample):
    global objCascade
    imgColor = cv2.imread(Sample)
    if imgColor.shape[0] > 512:
        SizeY = int(imgColor.shape[1]*0.5)
        SizeX = int(imgColor.shape[0]*0.5)
        imgResized = cv2.resize(imgColor,(SizeY,SizeX))
        cv2.imshow('frame', imgResized)
    else:
        cv2.imshow('frame', imgColor)
    let = cv2.waitKey(0)
    imgGray = cv2.cvtColor(imgColor.copy(), cv2.COLOR_BGR2GRAY)
    #print 'shape img: %s -> shape imgColor: %s' %(str(img.shape),str(imgColor.shape))

    # Detect faces in the image
    objRes = objCascade.detectMultiScale(
        imgGray,
        scaleFactor=1.02,
        minNeighbors=20,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    #print("Found {0} faces!".format(len(faces)))
    objCount = 0  

    #imgColored = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    imgColored = imgColor[:]     
   
    IndexIma = 0
    ImageListName = []
    for (x, y, w, h) in objRes:
        cv2.rectangle(imgColored, (x, y), (x+w, y+h), (0, 255, 0), 1)
        imgCrop = imgColor[y:y+h,x:x+w].copy()
        #imgCrop = imgColored[y:h,x:w]
        ImageListName += [str(IndexIma)+'.jpg']
        cv2.imwrite(ImageListName[IndexIma],imgCrop)
        IndexIma += 1

           
    return ImageListName

def GetDataLine(Data,Line):
    ToReturn = []
    for I in range(Data.shape[0]):
        ToReturn += [Data[Line][I]]
    return ToReturn

def CheckResistor(imgIn):
    imgGray = cv2.cvtColor(imgIn.copy(), cv2.COLOR_BGR2GRAY)

    imgGray = cv2.blur(imgGray,(12,12))
    imgGray = cv2.blur(imgGray,(12,12))
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(imgGray,cv2.CV_32F,1,0,-1)
    gradY = cv2.Sobel(imgGray,cv2.CV_32F,0,1,-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (12, 12))
    (_, thresh) = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    ThreshShape = thresh.shape

    LineIni = 0
    LineEnd = 0

    for i in range(ThreshShape[0]):
        DataX = GetDataLine(thresh,i)
        MaxiX = max(DataX)
        #print MaxiX
        if LineIni == 0:
            if MaxiX > 0:
                LineIni = i
        else:
            if LineEnd == 0:
                if MaxiX == 0:
                    LineEnd = i    
    if LineEnd - LineIni > 78:
        print('Size: ' + str(LineIni) + ' '+ str(LineEnd))
        return True
    else:
        print('Size: ' + str(LineIni) + ' '+ str(LineEnd))
        return False

ResistorsInImage = FindInImage(sys.argv[1])   
##        2, 16, 
print(ResistorsInImage)

imgSRC = ResistorsInImage[:]

for i in range(len(imgSRC)):
    print('Processing image: ' + imgSRC[i])
    imgOriginal = cv2.imread(imgSRC[i])
    imgOrigShape = imgOriginal.shape

    IsResistor = CheckResistor(imgOriginal)

    if IsResistor:
        bands = ColourSeparation.getColourBands(imgOriginal, save_blobs=True)
        ColourSeparation.getResistance(bands)