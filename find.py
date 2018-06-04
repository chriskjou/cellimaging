import classify
import cv2
from helpers import sliding_window, screenshot
import numpy as np
from helpers import truncate, sliding_window
import glob

### GLOBAL VARIABLES
window_size = 140
winW = 28
winH = 28
smallwindow_step = 28
radius = 70
smallwindowthreshold = 0.8

# files = ["test.jpg"]
# testcoords = {"test.jpg": [(400,200), (800,600)] }
# cellcounts = {}

# new_image = new_image[0:140, 0:140]
# cv2.imshow("test", new_image)

# heatmap = np.zeros((5,5,3))
def getcellcount(new_image):
    count = 0
    for (x, y, window) in sliding_window(new_image, stepSize=smallwindow_step, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        predictions = classify.isball(window)
        if predictions[0] >= smallwindowthreshold:
            count+=1
    return count

def foreachfile(endfolder, csvfolder):
    files = glob.glob(endfolder)
    csvfiles = glob.glob(csvfolder)

    cellcounts = {}
    # get cell centers
    for i in files:
        # append csv of centers #need to alter when more than one
        imgnamefirst = i.split(".")[0]
        imgname = imgnamefirst.split("/")[-1]
        cellcenters = screenshot(imgname)
        cellcounts.setdefault(i, [])
        img = cv2.imread(i)
        counts = []
        height, width, _ = img.shape
        for coord in cellcenters:
            xcoord = coord[0]
            ycoord = coord[1]

            lowerx = max(0, xcoord - radius)
            higherx = min(width, xcoord + radius)

            lowery = max(0, ycoord - radius)
            highery = min(height, ycoord + radius)

            checkimage = img[lowerx: higherx, lowery: highery]

            counts.append(getcellcount(checkimage))
        cellcounts[i] = counts

    cells = []
    for i in files:
        cells.extend(cellcounts[i])
    print("cellcounts")
    print(cellcounts)

    print("CELLS")
    print(cells)
    print(len(cells))

    return cells

print(foreachfile("boundary/*", "csv/*"))
