import cv2
import numpy as np
import argparse
import math
from skimage.morphology import reconstruction
import matplotlib.pyplot as plt
import glob

#== Parameters =======================================================================
BLUR = 21 #21
CANNY_THRESH_1 = 80 #10
CANNY_THRESH_2 = 115 #200
MASK_DILATE_ITER = 10 #10
MASK_ERODE_ITER = 10 #10
MASK_COLOR = (0.0,0.0,0.0) # In BGR format

def fillhole(img):
    seed = np.copy(img)
    seed[1:-1,1:-1] = np.max(img)
    mask= img
    return reconstruction(seed,mask,method='erosion').astype(np.uint8)

def getdropcenters(image, sizedict):
    height, width = sizedict[0], sizedict[1]
    #== Processing =======================================================================
    title = image.split(".")[0]
    #-- Read image -----------------------------------------------------------------------
    im = cv2.imread(image)
    im = fillhole(im)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    #-- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    #-- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # cv2.drawContours(im, contours, -1, (0, 0, 255), 3)
    # cv2.imshow("all contours", im)
    # cv2.waitKey(0)

    contours_area = []
    # calculate area and filter into new array
    for con in contours:
        area = cv2.contourArea(con)
        if 10000 < area < 25000:
            contours_area.append(con)

    # cv2.drawContours(im, contours_area, -1, (0, 255, 0), 3)
    # cv2.imshow("contour area", im)
    # cv2.waitKey(0)

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

    # cv2.drawContours(im, contours_circles, -1, (255, 0, 0), 3)
    # cv2.imshow("contours circles", im)
    # cv2.waitKey(0)

    # print("CONTOUR CIRCLS")
    # print("CONTOUR CIRCLES LENGTH: " + str(len(contours_circles)))
    centers = []
    bounds = []
    plt.imshow(im)
    for c in contours_circles:
        (x,y),radius = cv2.minEnclosingCircle(c)
        plt.plot(x, y, 'ro')
        centers.append((round(x),round(y),radius))

        lowerx = max(0, x - radius)
        higherx = min(width, x + radius)
        lowery = max(0, y - radius)
        highery = min(height, y + radius)
        bounds.append((lowerx, higherx, lowery, highery))

    plt.savefig("centers/centers.jpg")
    print("CENTERS IN GETCENTERS: ")
    print("LENGTH OF CENTERS: " + str(len(centers)))
    return centers, bounds

# getdropcenters("trial.tif", (1040, 1392))
