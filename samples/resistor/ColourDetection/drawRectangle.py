# Draw rectangles over images
import cv2

img_path = "C:\\Users\\Mloong\\Documents\\Code\\OrbitalProject\\Mask_RCNN_TF2_Compatible\\samples\\resistor\\single_resistor-mask0.png"

# variables
coordinates = []
drawing = False
coordsBackup = []
i = 0

img = cv2.imread(img_path)
start = img.copy()

#saves each drawing of an image separately
backup = []
backup.append(img.copy())
prevImg = backup[i].copy()

def draw_reactangle_with_drag(event, x, y, flags, param):
    global coordinates, drawing, prevImg, i
    #left click to start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        prevImg = img.copy()
        coordinates = [(x, y)]

    #release left click to stop drawing
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        coordinates.append((x, y))
        cv2.rectangle(img, pt1=coordinates[0], pt2=coordinates[1], color=(255,255,255), thickness=2)
        backup.append(img.copy())
        coordsBackup.append(coordinates)
        i += 1
    
    #display the shape while the mouse is still moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            prevImg = img.copy()
            cv2.rectangle(prevImg, pt1=coordinates[0], pt2=(x,y), color=(255,255,255), thickness=2)

cv2.namedWindow(winname= "image")
cv2.setMouseCallback("image", draw_reactangle_with_drag)

while True:
    if drawing:
        cv2.imshow("image", prevImg)
    else:
        cv2.imshow("image", img)
    
    key = cv2.waitKey(1) & 0xFF

    #quit
    if key == ord("q"):
        break

    #undo
    elif key == ord("r"):
        img = backup[i-1].copy()
        if i > 0:
            backup.pop()
            coordsBackup.pop()
            i -= 1
    
    #clear everything
    elif key == ord("c"):
        img = start.copy()
        prevImg = start.copy()
        coordsBackup = []

    #save image
    elif key == ord("s"):
        cv2.imwrite("boxed_images.png", img)
        print("Saving image with boxes")
        k = 0
        for topLeft, bottomRight in coordsBackup:
            print(topLeft, bottomRight)
            image = img[topLeft[1]+2:bottomRight[1]-2, topLeft[0]+2:bottomRight[0]-2]
            cv2.imwrite(f"cropped_image{k}.png", image)
            k += 1
            print("Saving individual boxes")

cv2.destroyAllWindows()