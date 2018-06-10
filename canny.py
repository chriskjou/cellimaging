import cv2
import numpy as np
import math
from skimage.morphology import reconstruction
import matplotlib.pyplot as plt
from datetime import datetime
import glob

def canny(image, CANNY_THRESH_1, CANNY_THRESH_2):
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	#-- Edge detection -------------------------------------------------------------------
	edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
	edges = cv2.dilate(edges, None)
	edges = cv2.erode(edges, None)

	#-- Find contours in edges, sort by area ---------------------------------------------
	contour_info = []
	_, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

	return contours, edges

def getcellnum(image):

	#== Parameters =======================================================================
	CANNY_THRESH_1 = 200 #10
	CANNY_THRESH_2 = 300 #200

	contours, _ = canny(image, CANNY_THRESH_1, CANNY_THRESH_2)

	# print("CONTOURS:" + str(len(contours)))

	contours_area = []
	for con in contours:
		area = cv2.contourArea(con)
		if 50 < area < 5000:
			contours_area.append(con)

	# cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
	# cv2.imshow("all contours", image)
	# cv2.waitKey(0)
	# plt.imshow(image)

	## UNCOMMENT BELOW FOR SAVED TEST IMAGE
	# count = 1
	# for con in contours_area:
	# 	(x,y),radius = cv2.minEnclosingCircle(con)
	# 	plt.plot(x, y, 'ro')
	# 	plt.text(x, y + 5, str(count))
	# 	count += 1
	# plt.savefig("adaptivecanny.jpg")
	# plt.clf()
	## UNCOMMENT ABOVE FOR SAVED TEST IMAGE

	# print("CONTOURS AREAS:" + str(len(contours_area)))

	contours_circles = []
	for con in contours_area:
		perimeter = cv2.arcLength(con, True)
		area = cv2.contourArea(con)
		if perimeter == 0:
			break
		circularity = 4*math.pi*(area/(perimeter*perimeter))
		if 0.5 < circularity and circularity < 1.5:
			contours_circles.append(con)

	# print("CONTOURS CIRCLES:" + str(len(contours_circles)))

	# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
	# cv2.imshow("all contours", image)
	# cv2.waitKey(0)

	return len(contours_area)


def preprocess(img):
	#== Parameters =======================================================================
	BLUR = 1 #21
	CANNY_THRESH_1 = 100 #10
	CANNY_THRESH_2 = 200 #200
	MASK_DILATE_ITER = 10 #10
	MASK_ERODE_ITER = 10 #10
	MASK_COLOR = (0.0,0.0,0.0) # In BGR format

	contour_info = []
	contours, edges = canny(img, CANNY_THRESH_1, CANNY_THRESH_2)

	for c in contours:
	    contour_info.append((
	        c,
	        cv2.isContourConvex(c),
	        cv2.contourArea(c),
	    ))
	contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
	max_contour = contour_info[0]

	#-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
	# Mask is black, polygon is white
	mask = np.zeros(edges.shape)
	cv2.fillConvexPoly(mask, max_contour[0], (255))

	#-- Smooth mask, then blur it --------------------------------------------------------
	mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
	mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
	#mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
	mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

	#-- Blend masked img into MASK_COLOR background --------------------------------------
	mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices,
	img         = img.astype('float32') / 255.0                 #  for easy blending

	masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
	masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit

	return getcellnum(masked)

def getbounds(image, bounds):
	img = cv2.imread(image)
	toreturn = []
	for bound in bounds:
		lowery, highery, lowerx, higherx = bound
		toreturn.append(preprocess(img[lowery: highery, lowerx: higherx]))
	return toreturn
