# -*- coding: utf-8 -*-
"""
Created on Thu May 31 09:49:17 2018

@author: Home
"""

import os
import cv2
import numpy as np

#test_dirs = os.listdir("images/")
#test_filenames=["images/"+file_id+"/"+file_id+".png" for file_id in test_dirs]
test_filenames = ["corner-512.png", "corner2-512.png", "corner3-512.png", "corner4-512.png"]
test_images=[cv2.imread(imagefile) for imagefile in test_filenames]

#for file_id in test_dirs:
#    print(file_id)

print(len(test_images))

def process(img_rgb):
    #green channel happends to produce slightly better results
    #than the grayscale image and other channels
    img_gray=img_rgb[:,:,1]#cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #morphological opening (size tuned on training data)
    circle7=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    img_open=cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, circle7)
    #Otsu thresholding
    img_th=cv2.threshold(img_open,0,255,cv2.THRESH_OTSU)[1]
    #Invert the image in case the objects of interest are in the dark side
    if(np.sum(img_th==255)>np.sum(img_th==0)):
        img_th=cv2.bitwise_not(img_th)
    #second morphological opening (on binary image this time)
    bin_open=cv2.morphologyEx(img_th, cv2.MORPH_OPEN, circle7) 
    #connected components
    cc=cv2.connectedComponents(bin_open)[1]
    #cc=segment_on_dt(bin_open,20)
    return cc

test_connected_components=[process(img)  for img in test_images]

def rle_encoding(cc):
    print(type(cc))
    print(cc)
    values=list(np.unique(cc))
    values.remove(0)
    RLEs=[]
    for v in values:
        dots = np.where(cc.T.flatten() == v)[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if (b>prev+1):
                run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        RLEs.append(run_lengths)
    return RLEs

test_RLEs=[rle_encoding(cc) for cc in test_connected_components]

with open("submission_image_processing-CHECK-512.csv", "a") as myfile:
    myfile.write("ImageId,EncodedPixels\n")
    for i,RLEs in enumerate(test_RLEs):
        for RLE in RLEs:
            myfile.write(test_filenames[i]+","+" ".join([str(i) for i in RLE])+"\n")
            
