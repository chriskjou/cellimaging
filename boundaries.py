import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import random
import warnings
import math
import gc
import numpy as np
import pandas as pd
import csv

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.morphology import reconstruction
import scipy.ndimage as ndimage
import cv2
import helpers as H
import tensorflow as tf
import glob
from PIL import Image
from getcenters import getdropcenters
from canny import getbounds
from classify import classifycellnum
from classify2 import classifyinfectedcellnum

def removeinfected(folder):
    actual = []
    for each in folder:
        temp = each.split(".")[0]
        if temp[-3] == '2':
            actual.append(each)
    return actual

def everything(imagefolder):
    is_training = False
    image_class = 0

    # Set some parameters
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3

    sizes_test = []
    filesdict = {}
    infectedperdropdict = {}
    imageids = []
    cellnum = []
    infected = []
    cells = []
    count = 1

    testfiles = glob.glob(imagefolder)
    testfiles = removeinfected(testfiles)

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    print('collecting the following images...')
    sizedict = {}

    for each in testfiles:
        print(each)
        img = cv2.imread(each)
        bw, img = H.imscale(img)
        if bw == image_class:
            sizes_test.append([img.shape[0], img.shape[1]])
            sizedict.setdefault(each, []).append((img.shape[0], img.shape[1]))

    print("\nreading ", len(sizes_test), " of ", len(testfiles))

    print("\npredicting based on the model...\n")

    for i in testfiles:
        infectedperdropdict.setdefault(i, [])
        print("classifying ", str(count), " of ", len(testfiles))
        count+=1

        # get file names
        temp = i.split(".")[0]
        ending = i.split(".")[1]
        search = temp[:-3] + "1t" + temp[-1] + "." + ending

        # get cell centers
        cellcoords, bounds = getdropcenters(search, sizedict[i][0], i)

        cells = []
        infected = []

        for bound in bounds:
            lowery, highery, lowerx, higherx = bound

            imgbright = cv2.imread(search)
            imginfect = cv2.imread(i)

            subbright = imgbright[lowery: highery, lowerx: higherx]
            subinfect = imginfect[lowery: highery, lowerx: higherx]

            cells.append(classifycellnum(subbright))
            infected.append(classifyinfectedcellnum(subinfect))

        cellcount = len(infected)
        imageids.extend([i] * cellcount)
        generatenums = [j for j in range(1, cellcount + 1)]
        cellnum.extend(generatenums)

    return imageids, cellnum, infected, cells
