import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import time
from collections import Counter

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
folder_path = '/Users/yukaaoyama/engi200/images'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
# initialize frame and image counter
frame_counter = 0

newFastener = True

def empty(a):
    pass

# Scroll bars for adjustments
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

# For display
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
        return None #?
    else:
        # print("not found contours")
        return None
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
                cv2.putText(imgContour, "Nut", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 0), 2)
                return True, 0
                # print("Nut")
            else:
                cv2.putText(imgContour, "Washer", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 0), 2)
                return True, 1
                # print("Washer")
        # return True
    else:
        return None, -1
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
        # Need to make sure slope is defined (undefined = vertical)
        # if second_max_coord[0]-max_coord[0] == 0.0:

        # if second_min_coord[0]-min_coord[0] == 0.0:
            
        slope_max = (second_max_coord[1]-max_coord[1]) / (second_max_coord[0]-max_coord[0]) 
        slope_min = (second_min_coord[1]-min_coord[1]) / (second_min_coord[0]-min_coord[0]) 

        if slope_max == 0.0:
            if abs(slope_min) < 0.5:
                cv2.putText(imgContour, "Bolt", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 0), 2)
                return 3
                # print("Bolt")
        elif slope_min == 0.0:
            if abs(slope_max) < 0.5:
                cv2.putText(imgContour, "Bolt", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 0), 2)
                return 3
                # print("Bolt")
        else:
            #use ratio
            slope_ratio = abs(slope_max / slope_min)
            # print(slope_ratio)
            if 0.4 <= slope_ratio <= 2:
                cv2.putText(imgContour, "Bolt", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 0), 2)
                return 3
                # print("Bolt")
            else:
                cv2.putText(imgContour, "Screw", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 0), 2)
                return 4
                # print("Screw")

while True:
    success, img = cap.read()
    frame_counter += 1

    imgContour = img.copy()
    imgRange = img.copy()
    imgResult = img.copy()

    # Image Processing
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

        # Run classification algorithms only when the fastner is in range
        if x>=220 and x<=420:
            # Create a new list for collecting results only if the fastener has just entered
            if newFastener is True:
                results = []
            newFastener = False

            # Save frame as image in specified folder
            if frame_counter % 1 == 0: #this camera is 7.50fps
            #     print("pic")
            #     filename = os.path.join(folder_path, f'image_{img_counter}.jpg')
                # filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.jpg'
            #     # print("writing")
            #     cv2.imwrite(filename, img)

                isWasherOrNut, fastener = washerOrNut(imgDil, contours, x, y, w, h)

                if isWasherOrNut is None:
                    fastener = screwOrBolt(imgDil)

                if fastener is not None:
                    results.append(fastener)
                # print(results)

        # Case if fastener is not within the range we are looking at (haven't looked at or finished looking at previous)
        else:
            # Results from the previous one
            final_result = ""
            if newFastener is False:
                counter = Counter(results)
                majority = counter.most_common(1)
                if majority[0][0] == 0:
                    final_result = "Nut"
                elif majority[0][0] == 1:
                    final_result = "Washer"
                elif majority[0][0] == 2:
                    final_result = "Bolt"
                elif majority[0][0] == 3:
                    final_result = "Screw"

            if final_result != "":
                print(final_result)
            cv2.putText(imgResult, final_result, (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 0), 2)

            #next fastener that comes in the range is a new fastener
            newFastener = True 
            #QUESTION: Why does it quit?

    imgStack = stackImages(0.8, ([img, imgGray, imgCanny], [imgRange, imgContour, imgResult]))

    cv2.imshow("Results", imgStack)
    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break