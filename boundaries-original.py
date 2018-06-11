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
from skimage import img_as_float, exposure, img_as_ubyte
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.morphology import label
from keras.models import Model, load_model, model_from_json
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
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

    testfiles = glob.glob(imagefolder)
    # testfiles = removeinfected(testfiles)
    print("TESTFILES: " + str(testfiles))

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    seed = 42
    random.seed = seed
    np.random.seed = seed

    trainlist = []

    # store samples that are not in class needed
    non_ids = []

    Xtest = []
    sizes_test = []
    shape_test = []
    scale_test = [0]
    non_ids = []
    print('getting and resizing test images ... ')
    sys.stdout.flush()
    sizedict = {}

    for each in testfiles:
        print(each)
        ending = each.split(".")[-1]
        img = ""
        if ending == "tiff" or ending == "tif":
            imgpath = each.split(".")
            imgtype = imgpath[-1]
            grayscale = Image.open(each).convert('L')
            grayscale.save('temp.png', "PNG")
            png = cv2.imread('temp.png')
            gspng = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
            torgb = cv2.cvtColor(gspng, cv2.COLOR_GRAY2BGR)
            img = torgb[:,:,:IMG_CHANNELS]
        elif ending == "png":
            img = cv2.imread(each)[:,:,:IMG_CHANNELS]
        else:
            print("ERROR: make sure all images are tiff or pngs")
            return 0
        bw, img = H.imscale(img)
        if bw == image_class:
            sizes_test.append([img.shape[0], img.shape[1]])
            sizedict.setdefault(each, []).append((img.shape[0], img.shape[1]))
            shape_test.append([img.shape])
        else:
            non_ids.append(each)
        imgs = H.img_sample(img)
        if bw == image_class:
            scale_test.append(scale_test[-1] + len(imgs))
            for img in imgs:
                Xtest.append(img)

    X_test = np.zeros((len(Xtest), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    for i, ximg in enumerate(Xtest):
        X_test[i] = H.fcontrast(ximg)

    print('reading ', len(sizes_test), ' of ', len(testfiles))

    print("reading previously saved model...\n")
    json_file = open('models/mmodel'+str(image_class)+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("models/mmodel"+str(image_class)+".h5")

    print("predicting based on the model...\n")

    preds_test = model.predict(X_test, verbose=1)

    # Threshold predictions
    preds_test_t = (preds_test > 0.7).astype(np.uint8)

    # convert resampled test cases to original scale
    preds_test_a = []
    for i in range(len(shape_test)):
        img = H.img_assembly(preds_test_t[scale_test[i]:scale_test[i+1]],shape_test[i][0])
        preds_test_a.append(img)

    preds_test_upsampled = []
    for i in range(len(preds_test_a)):
        preds_test_upsampled.append(preds_test_a[i])

    new_test_ids = []
    rles = []
    files = []
    filesdict = {}
    infectedperdropdict = {}

    index = 0
    for each in testfiles:
        rle = list(H.prob_to_rles(preds_test_upsampled[index]))
        rles.extend(rle)
        new_test_ids.extend([each] * len(rle))
        files.append(each)
        infectedperdropdict.setdefault(each, [])
        filesdict.setdefault(each, [])
        index += 1

    with open('bwtrain'+str(image_class)+'.log', 'w') as myfile:
        wr = csv.writer(myfile)
        for bw in trainlist:
            wr.writerow([bw])

    with open('bwtrain'+str(image_class)+'.log', 'w') as myfile:
        wr = csv.writer(myfile)
        for bw in trainlist:
            wr.writerow([bw])

    im = cv2.imread(files[0])

    for i in range(len(rles)):
        name = new_test_ids[i]
        center = H.centerofrle(rles[i], sizedict[name][0][0], sizedict[name][0][1])
        filesdict[name].append(center)

    imageids = []
    cellnum = []
    infected = []
    cells = []
    count = 1
    # for i in files:
    #     print("classifying ", str(count), " of ", len(testfiles))
    #     count+=1
    #
    #     # get file names
    #     temp = i.split(".")[0]
    #     ending = i.split(".")[1]
    #     search = temp[:-3] + "1t" + temp[-1] + "." + ending
    #
    #     # get cell centers
    #     cellcoords, bounds = getdropcenters(search, sizedict[i][0], i)
    #     cells.extend(classifycellnum(search, bounds))
    #
    #     # ONLY CHOOSE ONE OF THE FOLLOWING THREE TO UNCOMMENT
    #     ### (1)
    #     # with cannny threshold
    #     # cells.extend(getbounds(search, bounds))
    #
    #     ### (2)
    #     # with google inception v3
    #     infectedcount = classifyinfectedcellnum(i, bounds)
    #     infectedperdropdict[i] = infectedcount
    #     infected.extend(infectedcount)
    #
    #     ### (3)
    #     # using keras model
    #     # infectedperdropdict should be a dictionary
    #     # key: each image
    #     # value: dictionary where key is cell # and value is infected cell count
    #     # infectedperdropdict[i] = H.updateDict(filesdict[i], cellcoords, bounds)
    #     # infected.extend(infectedperdropdict[i])
    #     # ONLY CHOOSE ONE OF THE ABOVE THREE TO UNCOMMENT
    #
    #     # updating count
    #     cellnums = len(infectedperdropdict[i])
    #     imageids.extend([i] * cellnums)
    #     generatenums = [j for j in range(1, cellnums + 1)]
    #     cellnum.extend(generatenums)

    # UNCOMMENT BELOW FOR CSV for rle encoding with total number per image
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    counts = sub['ImageId'].value_counts()
    uniqueids = sub['ImageId'].unique()
    cellcounter = []
    for i in uniqueids:
        cellcounter.append(counts[i])
    actual = pd.DataFrame()
    actual["IMAGE"] = uniqueids
    actual["count"] = cellcounter
    # sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x)
    actual.to_csv('cellcountwellplate.csv', index=False)
    print("Done adding cell counts to CSV.")
    # UNCOMMENT ABOVE FOR CSV for rle encoding with total number per image

    return imageids, cellnum, infected, cells

print(everything("wellplate/*"))
