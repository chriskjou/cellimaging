### WORKS ON TIFS ###
import classify
import cv2
from helpers import sliding_window, screenshot
import numpy as np
from helpers import truncate
import glob

### GLOBAL VARIABLES
window_size = 140
winW = 28
winH = 28
smallwindow_step = 28
radius = 70
smallwindowthreshold = 0.8

def getcellcount(new_image):
    count = 0
    for (x, y, window) in sliding_window(new_image, stepSize=smallwindow_step, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        predictions = classify.isball(window)
        if predictions[0] >= smallwindowthreshold:
            count+=1
    return count

### ALTERED HERE ###
def foreachfile(endfolder, infectfolder, csvfolder, sizedict):
    files = glob.glob(endfolder)
    start = infectfolder.split("/")[-2]
    print("START: " + str(start))
    csvfiles = glob.glob(csvfolder)

    cellcounts = {}

    for i in files:
        temp = i.split(".")
        imgnamefirst = temp[0]
        ending = temp[1]
        imgname = imgnamefirst.split("/")[-1]
        print("IMGNAME: " + str(imgname))
        search = start + "/" + imgname + "." + ending
        cellcenters, bounds = screenshot(imgname, sizedict[search][0])
        cellcounts.setdefault(i, [])
        img = cv2.imread(i)
        counts = []
        height, width, _ = img.shape
        for bound in bounds:
            lowerx, higherx, lowery, highery = bound
            # xcoord = coord[0]
            # ycoord = coord[1]
            #
            # lowerx = max(0, xcoord - radius)
            # higherx = min(width, xcoord + radius)
            #
            # lowery = max(0, ycoord - radius)
            # highery = min(height, ycoord + radius)

            checkimage = img[lowerx: higherx, lowery: highery]

            counts.append(getcellcount(checkimage))
        cellcounts[i] = counts

    cells = []
    for i in files:
        cells.extend(cellcounts[i])
    return cells
### ALTERED HERE ###
# print(foreachfile("imagetif/*", "csvs/*"))
