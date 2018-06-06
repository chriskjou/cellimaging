# OpenCV blob detection
# https://www.learnopencv.com/blob-detection-using-opencv-python-c/

import cv2
import numpy as np

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 220
# params.thresholdStep = 10

# Filter by Area.
params.filterByArea = True
params.minArea = 400
params.maxArea = 2900

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.01

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

# Read image
im = cv2.imread("images.png")
# im = cv2.imread("threshold.jpg" , cv2.IMREAD_GRAYSCALE)

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# print(keypoints)
# Show keypoints
cv2.imshow("blob detector", im_with_keypoints)
cv2.imwrite("blobdetector.png", im_with_keypoints)
cv2.waitKey(0)
