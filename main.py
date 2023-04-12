import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import time

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",50,255,empty)
cv2.createTrackbar("Threshold2","Parameters",100,255,empty)
cv2.createTrackbar("Low H","Parameters",40,255,empty)
cv2.createTrackbar("Low S","Parameters",0,255,empty)
cv2.createTrackbar("Low V","Parameters",0,255,empty)
cv2.createTrackbar("High H","Parameters",83,255,empty)
cv2.createTrackbar("High S","Parameters",255,255,empty)
cv2.createTrackbar("High V","Parameters",255,255,empty)
cv2.createTrackbar("Area","Parameters",1000,30000,empty)

def save_all_frames(x, dir_path, basename, ext='jpg'):

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    starttime = time.time()

    # while True:
    #     if x>=220 and x<=420:
            # ret, frame = cap.read()
            # print('pic!')
            # time.sleep(60.0 - ((time.time() - starttime) % 60.0))
            # if ret:
            #     cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            #     n += 1
            # else:
            #     return
        # else: 
        #     return

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img,imgContour):
    # print("getting contours")
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #TODO can play around with the last two

    if contours is not None:
        for cnt in contours:
            # print("cnt",cnt)
            area = cv2.contourArea(cnt)
            areaMin = cv2.getTrackbarPos("Area", "Parameters")
            if area > areaMin:
                # print("area is larger")
                cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                # print(approx)
                # print(len(approx))
                x , y , w, h = cv2.boundingRect(approx)
                # print(img[x:x+w, y:y+h])
                cv2.imshow("Cropped", img[y-10:y+h+20, x-10:x+w+20])
                # print(x, y, w, h)
                cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)
                return contours, x, y, w, h
        #do something when contours is done because it will quit
    else:
        # print("not found contours")
        return False
    # print("end of find contours")

def removeBackground(imgBlur):
    hsv = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)

    # Create a Green Mask
    low_h = cv2.getTrackbarPos("Low H", "Parameters")
    low_s = cv2.getTrackbarPos("Low S", "Parameters")
    low_v = cv2.getTrackbarPos("Low V", "Parameters")
    high_h = cv2.getTrackbarPos("High H", "Parameters")
    high_s = cv2.getTrackbarPos("High S", "Parameters")
    high_v = cv2.getTrackbarPos("High V", "Parameters")
    low_green = np.array([low_h, low_s, low_v])
    high_green = np.array([high_h, high_s, high_v])
    mask = cv2.inRange(hsv, low_green, high_green)

    # Convert Green to White
    hsv[mask>0]=(0,0,255) 

    # Convert Everything Else to Black
    mask = 255-mask
    hsv[mask>0]=(0,0,0)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Alternative option: Remove everything outside of the bounding box
# def cleanBackground(img, x, y, w, h):
#     #How?

def washerOrNut(img, contours, x, y, w, h):
    # imgFastener = img.copy()
    # cv2.imshow("washer or nut", img[y-10:y+h+20, x-10:x+w+20])
    detected_circles = cv2.HoughCircles(img[y-10:y+h+20, x-10:x+w+20], 
                   cv2.HOUGH_GRADIENT, 1, 0.01, param1 = 200,
               param2 = 20, minRadius = 8, maxRadius = 150)

    if detected_circles is not None:
        # print("circle was detected")
        # thresh = cv2.threshold(img[x-10:x+w+20, y-10:y+h+20], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # # Find contours and detect shape
        # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 6:
                cv2.putText(imgContour, "Nut", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 0), 2)
                # print("Nut")
            else:
                cv2.putText(imgContour, "Washer", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 0), 2)
                # print("Washer")
        return True
    # else:
    #     return False
    

def screwOrBolt(img):
    corners = cv2.goodFeaturesToTrack(img[y-10:y+h+20, x-10:x+w+20], 20, 0.01, 5, useHarrisDetector=True, k=0.01)
    if corners is not None:
        corners = np.int0(corners) #float to integer

        max_val = float('-inf')
        max_coord = 0
        second_max_val = float('-inf')
        min_val = float('inf')
        min_coord = 0
        second_min_val = float('inf')

        for thing in corners:
            if thing[0][1] > max_val:
                second_max_val = max_val
                second_max_coord = max_coord
                max_val = thing[0][1]
                max_coord = thing[0]
            elif thing[0][1] > second_max_val:
                second_max_val = thing[0][1]
                second_max_coord = thing[0]
            if thing[0][1] < min_val:
                second_min_val = min_val
                second_min_coord = min_coord
                min_val = thing[0][1]
                min_coord = thing[0]
            elif thing[0][1] < second_min_val:
                second_min_val = thing[0][1]
                second_min_coord = thing[0]
        
        # cv2.circle(imgCorner, max_coord, 3, 255, -1)
        # cv2.circle(imgCorner, second_max_coord, 3, 255, -1)
        # cv2.circle(imgCorner, min_coord, 3, 255, -1)
        # cv2.circle(imgCorner, second_min_coord, 3, 255, -1)

        #use slope
        slope_max = (second_max_coord[1]-max_coord[1]) / (second_max_coord[0]-max_coord[0]) 
        slope_min = (second_min_coord[1]-min_coord[1]) / (second_min_coord[0]-min_coord[0]) 

        if slope_max == 0.0:
            if abs(slope_min) < 0.5:
                cv2.putText(imgContour, "Bolt", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                            (0, 255, 0), 2)
                # print("Bolt")
        elif slope_min == 0.0:
            if abs(slope_max) < 0.5:
                cv2.putText(imgContour, "Bolt", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                            (0, 255, 0), 2)
                # print("Bolt")
        else:
            #use ratio
            slope_ratio = abs(slope_max / slope_min)
            # print(slope_ratio)
            if 0.4 <= slope_ratio <= 2:
                cv2.putText(imgContour, "Bolt", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                            (0, 255, 0), 2)
                # print("Bolt")
            else:
                cv2.putText(imgContour, "Screw", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                            (0, 255, 0), 2)
                # print("Screw")

while True:
    success, img = cap.read()
    imgContour = img.copy()
    imgRange = img.copy()

    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)

    imgNoBackground = removeBackground(imgBlur)

    imgGray = cv2.cvtColor(imgNoBackground, cv2.COLOR_BGR2GRAY)

    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray,threshold1,threshold2)

    #Remove overlapped lines, which might not be necessary for our project. Can decide by testing the classifications later. TODO
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    # newFastner = True

    # getContours(imgDil, imgContour)
    if getContours(imgDil, imgContour) is not None:
        contours, x, y, w, h = getContours(imgDil, imgContour) #x, y is top left corner

        cv2.rectangle(imgRange, (200 , 0 ), (420 , 480 ), (0, 255, 0), 5)

        save_all_frames(x, '/Users/yukaaoyama/engi200/images', 'sample_img')

        # Run classification algorithms only when the fastner is in range
        # if newFastner is False:
        #     output_hist = plt.hist(fastner_type, bins='auto')
        if x>=220 and x<=420:
            isWasherOrNut = washerOrNut(imgDil, contours, x, y, w, h)

            if isWasherOrNut is None:
                screwOrBolt(imgDil)
        else:
            print("out of range")
            #QUESTION: Why does it quit?

    imgStack = stackImages(0.8, ([img, imgGray, imgCanny], [imgDil, imgContour, imgRange]))

    cv2.imshow("Results", imgStack)
    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break