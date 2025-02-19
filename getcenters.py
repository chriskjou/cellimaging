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
            x,y,r = temp[c]
            radius = r /math.sqrt(2)
            plt.plot(x, y, 'ro')
            centers.append((round(x),round(y),round(radius)))

            lowerx = max(0, round(x - radius))
            higherx = min(width, round(x + radius))
            lowery = max(0, round(y - radius))
            highery = min(height, round(y + radius))
            bounds.append((lowery, highery, lowerx, higherx))

            plt.text(x, y + 5, str(count))
            count +=1
    # print("BOUNDS: " + str(bounds))
    
    directory = "centers"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig("centers/" + name + "centers" + ".jpg")
    plt.clf()
    return centers, bounds
