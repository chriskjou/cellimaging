import cv2
import numpy as np
import argparse
import math
from skimage.morphology import reconstruction
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from canny import canny

#== Parameters =======================================================================
# BLUR = 21 #21
CANNY_THRESH_1 = 50 #10
CANNY_THRESH_2 = 115 #200
MASK_DILATE_ITER = 10 #10
MASK_ERODE_ITER = 10 #10
MASK_COLOR = (0.0,0.0,0.0) # In BGR format

def fillhole(img):
    seed = np.copy(img)
    seed[1:-1,1:-1] = np.max(img)
    mask= img
    return reconstruction(seed,mask,method='erosion').astype(np.uint8)

def getdropcenters(image, sizedict, entire):
    temp = entire.split("/")[-1]
    name = temp.split(".")[0]
    height, width = sizedict[0], sizedict[1]
    #== Processing =======================================================================
    title = image.split(".")[0]
    #-- Read image -----------------------------------------------------------------------
    im = cv2.imread(image)
    im = fillhole(im)
    contours, _ = canny(im, CANNY_THRESH_1, CANNY_THRESH_2)
    # gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # #-- Edge detection -------------------------------------------------------------------
    # edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    # edges = cv2.dilate(edges, None)
    # edges = cv2.erode(edges, None)
    #
    # #-- Find contours in edges, sort by area ---------------------------------------------
    # contour_info = []
    # _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contours_area = []
    # calculate area and filter into new array
    for con in contours:
        area = cv2.contourArea(con)
        if 10000 < area < 25000:
            contours_area.append(con)

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

    for c in range(length):
        if actual[c] == 1:
            x,y,radius = temp[c]
            plt.plot(x, y, 'ro')
            centers.append((round(x),round(y),round(radius)))

            lowerx = max(0, x - radius)
            higherx = min(width, x + radius)
            lowery = max(0, y - radius)
            highery = min(height, y + radius)
            bounds.append((round(lowery), round(highery), round(lowerx), round(higherx)))

            plt.text(x, y + 5, str(count))
            count +=1

    plt.savefig("centers/" + name + "centers" + ".jpg")
    plt.clf()
    return centers, bounds
