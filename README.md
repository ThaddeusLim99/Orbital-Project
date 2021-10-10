# Android app

This is an implementation of an android app tutorial that takes a picture and displays it in the app, with the camera function code done by Atif Pervaiz on youtube(https://www.youtube.com/watch?v=LpL9akTG4hI). This app uses Chaquopy, a method of using python in android studio as a starting point to insert the python code to detect the colours of the resistor bands into the app(https://chaquo.com/chaquopy/). The python code is taken from the mrcnntf2 branch with some edits to make it commpatible in android studio. The app language is java. 

This is updated version of the app. The original app can be found in the androidapp branch of the repository. This updated version does away with the mask-RCNN object detection method due to the intense load it places on the app. Instead, users will crop the image to find the resistor for the app to process. Through this method, app size has decreased significantly to over 120mb and processing time has also dropped dramatically from up to 2 minutes to up to 2 seconds. The downside is of course it sacrifices usability and some efficacy.

The image crop library is taken from https://github.com/Yalantis/uCrop.

See the PDF file named "Orbital milestone 3" for more imformation.

# Notes

1. The app is actually not in its most optimised state due to the use of Chaquopy. App may be able to be reduced in size further once code is converted to java. 
2. This app is using a free  and approved unlicensed version of Chaquopy. As stated in the commpany's website, the app will only work for 5 minutes from the point of activation of the app. Users will thus have to force stop the app in their phone settings to reactivate the app for further use.
3. Python code within this branch may or may not be up to date with the mrcnntf2 branch.

# Requirements

Android Studio and an android device is required. At least 120mb of space on your mobile device.

# Instructions

Download the zip file and extract the files inside. Open in Android Studio. Change the python path in the gradle file. Run on android device.

# Project poster
![Project poster](assets/presentation_poster.png)

# App flowchart

![App flowchart](assets/app_flowchart.png)

# App front page
![App Frontpage](assets/homepage.jpeg)

# App request of permissions
![App request of permissions](assets/requestingpermissions.jpeg)

# App choice of image
![Where to get the image from](assets/wheretogetimage.jpeg)

# App cropping UI
![App cropping UI](assets/croppingpage.jpeg)

# Cropped image
![Cropped image](assets/croppedimage.jpeg)

# App result page (correct identification)
![App Result page (correct identification)](assets/correctresult.jpeg)

# App result page (wrong identification)
![App result page (wrong identification)](assets/wrongresult.jpeg)

# App result page (when there is error)
![App result page (when there is error)](assets/detectionerror.jpeg)

# App feedback form
![App feedback form](assets/feedbackform.jpeg)

# App resistor code reference
![App resistor code reference](assets/resistorcode.jpeg)

# App manual resistor calculator
![App manual resistor calculator](assets/manual.jpeg)

