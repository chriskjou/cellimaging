import cv2
import numpy as np
import argparse
import math
from skimage.morphology import reconstruction
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from canny import canny
import os

#== Parameters =======================================================================
# BLUR = 21 #21
CANNY_THRESH_1 = 75 #10
CANNY_THRESH_2 = 150 #115 #200
MASK_DILATE_ITER = 10 #10
MASK_ERODE_ITER = 10 #10
MASK_COLOR = (0.0,0.0,0.0) # In BGR format

def fillhole(img):
    seed = np.copy(img)
    seed[1:-1,1:-1] = np.max(img)
    mask= img
    return reconstruction(seed,mask,method='erosion').astype(np.uint8)

def getdropcenters(image):
    #-- Read image -----------------------------------------------------------------------
    im = cv2.imread(image)
    im = fillhole(im)
    contours, _ = canny(im, CANNY_THRESH_1, CANNY_THRESH_2)

    # cv2.drawContours(im, contours, -1, (0, 0, 255), 3)
    # cv2.imshow("all contours", im)
    # cv2.waitKey(0)

    # print("CONTOURS:" + str(len(contours)))

    contours_area = []
    # calculate area and filter into new array
    for con in contours:
        area = cv2.contourArea(con)
        if 20 < area < 1000:
            contours_area.append(con)

    # cv2.drawContours(im, contours, -1, (0, 255, 0), 3)
    # cv2.imshow("contours area", im)
    # cv2.waitKey(0)

    # print("CONTOURS AREAS:" + str(len(contours_area)))

    contours_circles = []

    # check if contour is of circular shape
    for con in contours_area:
        perimeter = cv2.arcLength(con, True)
        area = cv2.contourArea(con)
        if perimeter == 0:
            break
        circularity = 4*math.pi*(area/(perimeter*perimeter))
        if 0.6 < circularity < 1.4:
            contours_circles.append(con)

    # centers = []
    # bounds = []
    # plt.imshow(im)
    # for c in contours_circles:
    #     (x,y),radius = cv2.minEnclosingCircle(c)
    #     plt.plot(x, y, 'ro')

    centers = []
    bounds = []
    plt.imshow(im)
    temp = []
    for c in contours_circles:
        (x,y),radius = cv2.minEnclosingCircle(c)
        temp.append((x,y,radius))

    count = 1
    length = len(contours_circles)
    actual = [1] * len(contours_circles)
    for i in range(length):
        x,y,r = temp[i]
        for j in range(i+1, length):
            x1, y1, r2 = temp[j]
            if abs(x-x1) < r and abs(y-y1) < r:
                if actual[j] == 1:
                    actual[j] = 0

    final = []
    for c in range(length):
        if actual[c] == 1:
            final.append(contours_circles[c])
            x,y,r = temp[c]
            radius = r /math.sqrt(2)
            plt.plot(x, y, 'ro')
            centers.append((round(x),round(y),round(radius)))

            plt.text(x, y + 5, str(count))
            count +=1

    print(len(centers))

    cv2.drawContours(im, final, -1, (255, 0, 0), 3)
    cv2.imshow("contours circles", im)
    cv2.waitKey(0)

    print("CONTOURS CIRCLES:" + str(len(final)))

    directory = "centers"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig("centers/" + datetime.now().strftime('%Y-%m-%d=%H-%M-%S') + "centers" + ".jpg")
    plt.clf()
#
# files = glob.glob("wellplate/*")
# for each in files:
#     print("EACH: " + each)
# print(getdropcenters("t0_b0s0c0x2183-1920y381-1216m0.tiff"))
print("NUM1:")
print(getdropcenters("wellplateneutral/wellplateneutralizationt1monoxy01c1t1.tif"))
print("NUM2:")
print(getdropcenters("wellplateneutral/wellplateneutralizationt1monoxy02c1t1.tif"))
print("NUM3:")
print(getdropcenters("wellplateneutral/wellplateneutralizationt1monoxy03c1t1.tif"))
