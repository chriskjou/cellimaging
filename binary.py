import cv2
import pylab
from scipy import ndimage
import argparse
import csv
import glob
files = glob.glob("MOI1/*")
print(files)

actual = []
for each in files:
    actual.append(each[5:])

def addtoCSV(images):
    f = open('MOI1binary.csv','w')
    columnTitleRow = "file, count\n"
    f.write(columnTitleRow)
    index = 0
    for i in images:
        f.write(i + ", " + str(getCount(i)) + "\n")
        index += 1
    f.close()

def getCount(im):
    im = cv2.imread(im)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    maxValue = 255
    adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C #cv2.ADAPTIVE_THRESH_MEAN_C
    thresholdType = cv2.THRESH_BINARY #cv2.THRESH_BINARY_INV
    blockSize = 5 # odd number
    C = -3 # constant to be subtracted
    im_thresholded = cv2.adaptiveThreshold(gray, maxValue, adaptiveMethod, thresholdType, blockSize, C)
    labelarray, particle_count = ndimage.measurements.label(im_thresholded)
    return particle_count

#addtoCSV(files)

#print("Finished writing to CSV...")

