import csv
import cv2
import os

# returns x,y location of center of the cell
def screenshot(refpath):
    flag = False
    # path = refpath + 'subimages/'
    #
    # if not os.path.exists(path):
    #     os.makedirs(path)

    # assuming that all cells are about 180 x 180 pixels

    # STEP 1: mark centers with imagej using macro
    # STEP 2: save centers to csv
    # STEP 3: get centers from csv
    print("finding " + refpath + ".csv ...")

    tofind = "csvs/" + refpath + ".csv"
    centers = []
    with open(tofind) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if not flag:
                flag = True
            else:
                xcoord = int(row[0])
                ycoord = int(row[1])
                centers.append((xcoord, ycoord))

    return centers
    # print("Finished transferring centers...")
    # # STEP 4: open flurescent image
    # img = cv2.imread("python.png")
    #
    # height = img.shape[0]
    # width = img.shape[1]
    #
    # # print("HW: " + str(height) + " by " + str(width))
    #
    # count = 0
    #
    # for center in centers:
    #     # print(center)
    #     xcenter = center[0]
    #     ycenter = center[1]
    #     # print("CENTERS: " + str(xcenter) + " " + str(ycenter))
    #
    #     lowerx = max(0, xcenter - cellradius)
    #     higherx = min(width, xcenter + cellradius)
    #
    #     lowery = max(0, ycenter - cellradius)
    #     highery = min(height, ycenter + cellradius)
    #
    #     # print(str(lowerx) + " " + str(higherx) + ", " + str(lowery) + " " + str(highery))
    #
    #     cv2.imwrite(os.path.join(path , str(count) + ".png"), img[lowery: highery, lowerx: higherx])
    #     count += 1
    #
    # print("Finished saving subimages...")
