import csv
from math import floor
import matplotlib.pyplot as plt
import cv2

# no cutoff
def centerofrle(rle, height, width):
    toprint = []
    area = height * width
    xrows = []
    ycolumns = []
    for index in range(0, len(rle), 2):
        pixelloc = int(rle[index])
        pixelnum = int(rle[index+1])
        if pixelloc <= area:
            toprint.append(pixelloc)
            for each in range(pixelnum):
                centerpixel = pixelloc + each
                xto = (centerpixel - 1) % height + 1
                xrows.append(xto)
                yto = floor((centerpixel - 1)/height) + 1
                ycolumns.append(yto)
        else:
            break
    midx = sum(xrows)/len(xrows)
    midy = sum(ycolumns)/len(ycolumns)
    return midy, midx

def readcsv(info):
    centers = []
    flag = False
    with open(info) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            # don't read in title
            if not flag:
                flag = True
            else:
                ## set width height per image
                width = 1392
                height = 1040
                encodedpixels = row[1]
                encodedpixels = encodedpixels.split(" ")
                center = centerofrle(encodedpixels, width, height)
                centers.append(center)
    return centers

def updateDict(centercoords, cellcoords, height, width, cellradius = 70):
    img = cv2.imread("alteredimages.png")
    plt.imshow(img)
    print("INSIDE UPDATE DICT")
    print("CENTERCOORDS: " + str(centercoords))

    print("CELLCOORDS: " + str(cellcoords))

    print("WIDTH: " + str(width))

    print("HEIGHT: " + str(height))
    # temp dictionary assigning infected cells to each
    # templist = [i for i in range(1, len(cellcoords)+ 1)]
    # infectedperdropdict = dict.fromkeys(templist)
    #
    # print("INITIALIZING INFECTED DICTIONARY")
    # print(infectedperdropdict)
    infectedlist = []

    for each in cellcoords:
        count = 0
        lowerx = max(0, each[0] - cellradius)
        higherx = min(width, each[0] + cellradius)

        lowery = max(0, each[1] - cellradius)
        highery = min(height, each[1] + cellradius)

        for coord in centercoords:
            xcoord = coord[0]
            ycoord = coord[1]

            if lowerx <= xcoord and xcoord <= higherx:
                if lowery <= ycoord and ycoord <= highery:
                    count += 1
                    plt.plot(xcoord, ycoord, 'ro')
        # infectedperdropdict[index] = count
        # index += 1
        infectedlist.append(count)
    plt.savefig('testlabels.jpg')
    # print(infectedperdropdict)
    # return infectedperdropdict
    return infectedlist
