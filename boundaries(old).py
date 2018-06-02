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

import matplotlib.pyplot as plt

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
from decoderle import centerofrle, readcsv, updateDict
from getscreenshot import screenshot

is_training = False
image_class = 0

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

TRAIN_PATH = 'stage1_train/'
# TEST_PATH = 'stage2_test/'
testfiles = glob.glob("boundary/*")
csvfiles = glob.glob("csv/*")

if len(testfiles) != len(csvfiles):
    print("ERROR: make sure there is a corresponding csv to each image")

# testfiles = glob.glob("005/*")
# testfiles = ["corner4-512.png", "corner3-512.png", "corner2-512.png", "corner-512.png"]

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

trainlist = []

# store samples that are not in class needed
non_ids = []

# get and resize train images and masks
XX_train = []
YY_train = []
sys.stdout.flush()
m = 0
if is_training:
    print('getting and resizing train images and masks ... ')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        bw, img = H.imscale(img)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            if np.sum(mask) == 0: # no prior mask available
                mask = mask_
            else:
                mask = np.maximum(mask, mask_)
        if bw == image_class :
            trainlist.append(id_)
            imgs, masks = H.img_resampled(img, mask)
            for img, mask in zip(imgs,masks):
                XX_train.append(img)
                YY_train.append(mask)
    #        XX_train[m] = img
    #        YY_train[m] = mask # img_as_float(mask)
            m += len(imgs)
        else:
            non_ids.append(id_)

    for untrain in non_ids:
        train_ids.remove(untrain)

    X_train = np.zeros((m, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((m, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    for i in range(m):
        X_train[i] = H.fcontrast(XX_train[i])
        Y_train[i] = YY_train[i]

    print('\n Read ',len(trainlist),'(%d)'%m,' samples of class',image_class,',',len(train_ids),' left')

    del XX_train, YY_train
    gc.collect()

Xtest = []
sizes_test = []
shape_test = []
scale_test = [0]
non_ids = []
print('getting and resizing test images ... ')
sys.stdout.flush()
sizedict = {}

if not is_training:

    for each in testfiles:
        print(each)
        # ending = each.split(".")[-1]
        # if ending == "tiff" or if ending == "tif":
        #     imgpath = each.split(".")
        #     imgtype = imgpath[-1]
        #     grayscale = Image.open(each).convert('L')
        #     grayscale.save('temp.png', "PNG")
        #     png = cv2.imread('temp.png')
        #     gspng = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
        #     torgb = cv2.cvtColor(gspng, cv2.COLOR_GRAY2BGR)
        #     img = torgb[:,:,:IMG_CHANNELS]
        img = cv2.imread(each)[:,:,:IMG_CHANNELS]

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

print("SIZEDICT")
print(sizedict)

print('reading ', len(sizes_test), ' of ', len(testfiles))

if is_training:
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[H.mean_iou])
    model.summary()
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model-dsbowl.h5', verbose=1, save_best_only=True)
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,
                        callbacks=[earlystopper, checkpointer])
else:
    print("reading previously saved model...\n")
    json_file = open('mmodel'+str(image_class)+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("mmodel"+str(image_class)+".h5")

print("predicting based on the model...\n")

if is_training:
    preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
    preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
if is_training:
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.7).astype(np.uint8)

# convert resampled test cases to original scale
preds_test_a = []
for i in range(len(shape_test)):
    img = H.img_assembly(preds_test_t[scale_test[i]:scale_test[i+1]],shape_test[i][0])
    preds_test_a.append(img)

preds_test_upsampled = []
for i in range(len(preds_test_a)):
    preds_test_upsampled.append(preds_test_a[i])

if is_training:
    # perform a sanity check on some random training samples
    ix = random.randint(0, len(preds_train_t))
    imshow(X_train[ix],cmap='gray')
    plt.show()
    imshow(np.squeeze(Y_train[ix]),cmap='gray')
    plt.show()
    imshow(np.squeeze(preds_train_t[ix]),cmap='gray')
    plt.show()

    # perform a sanity check on some random validation samples
    ix = random.randint(0, len(preds_val_t))
    imshow(X_train[int(X_train.shape[0]*0.9):][ix],cmap='gray')
    plt.show()
    imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]),cmap='gray')
    plt.show()
    imshow(np.squeeze(preds_val_t[ix]))
    plt.show()

# show predicted samples
# for i in range(len(preds_test_a)):
#     print(i,preds_test_a[i].shape,np.min(np.squeeze(preds_test_a[i])),np.mean(np.squeeze(preds_test_a[i])) )
#     imshow(np.squeeze(H.Intensifier(preds_test_upsampled[i])),cmap='gray')
#     plt.show()

new_test_ids = []
rles = []
files = []
filesdict = {}
infectedperdropdict = {}

### ADAPTED ###
index = 0
for each in testfiles:
    rle = list(H.prob_to_rles(preds_test_upsampled[index]))
    rles.extend(rle)
    new_test_ids.extend([each] * len(rle))
    files.append(each)
    infectedperdropdict.setdefault(each, [])
    filesdict.setdefault(each, [])
    index += 1

if is_training:
    model_json = model.to_json()
    with open("mmodel"+str(image_class)+".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("mmodel"+str(image_class)+".h5")
    print("saved model to disk...")

with open('bwtrain'+str(image_class)+'.log', 'w') as myfile:
    wr = csv.writer(myfile)
    for bw in trainlist:
        wr.writerow([bw])

with open('bwtrain'+str(image_class)+'.log', 'w') as myfile:
    wr = csv.writer(myfile)
    for bw in trainlist:
        wr.writerow([bw])

print("RLES LENGTH" + str(len(rles)))

print("NEW TEST IDS")
print(new_test_ids)
print("NEW TEST IDS LENGTH: " + str(len(new_test_ids)))

print("SIZEDICT")
print(sizedict)

print("FILES")
print(files)
for i in range(len(rles)):
    name = new_test_ids[i]
    center = centerofrle(rles[i], sizedict[name][0][0], sizedict[name][0][1])
    filesdict[name].append(center)

# gives infected as dictionary where key is image and value is center list
print("FILEDICT")
print(filesdict)

# create submission DataFrame
# sub = pd.DataFrame()
# sub['ImageId'] = new_test_ids
# sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
# sub.to_csv('sub-dsbowl'+str(image_class)+'.csv', index=False)

# centercoords = readcsv("sub-dsbowl0.csv")
# for index in range(len(rles)):
#     centercoords.append(centerofrle(rles[index], sizes_test[index][0], sizes_test[index][1]))
# # for each in rles:
# #     centercoords.append(centerofrle(each, 128,128))

# # droplet coordinates
# cellcoords = []
for i in files:
    # append csv of centers #need to alter when more than one
    # cellcoords.append(screenshot("testcenters"))
    imgnamefirst = i.split(".")[0]
    imgname = imgnamefirst.split("/")[-1]
    cellcoords = screenshot(imgname)
    infectedperdropdict[i] = updateDict(filesdict[i], cellcoords, sizedict[i][0][0], sizedict[i][0][1])
# cellcoords = screenshot("testcenters")

# infectedperdropdict should be a dictionary
# key: each image
# value: dictionary where key is cell # and value is infected cell count
print("INFECTEDPERDROPDICT")
print(infectedperdropdict)

imageids = []
cellnum = []
infected = []
for i in files:
    cellnums = len(infectedperdropdict[i])
    imageids.extend([i] * cellnums)
    generatenums = [j for j in range(1, cellnums + 1)]
    cellnum.extend(generatenums)
    infected.extend(infectedperdropdict[i])

sub = pd.DataFrame()
sub['ImageId'] = imageids
sub['Cell #'] = cellnum
sub['Infected'] = infected
sub.to_csv('trialboundaries'+str(image_class)+'.csv', index=False)

csvname = 'trialboundaris' + str(image_class)+'.csv'

print("finished exporting to " + csvname + "csv...")
# # temp dictionary assigning infected cells to each
# templist = [i for i in range(1, len(cellcoords)+ 1)]
# infectedperdropdict = dict.fromkeys(templist)
# print("INITIALIZING INFECTED DICTIONARY")
# print(infectedperdropdict)
#
# cellradius = 70
# width = 1392
# height = 1040
#
# print("CELL COORDS" + str(cellcoords))
# print("CENTER COORDS" + str(centercoords))
#
# # img = cv2.imread("alteredimages.png")
# # plt.imshow(img)
# index = 1
# for each in cellcoords:
#     count = 0
#     lowerx = max(0, each[0] - cellradius)
#     higherx = min(width, each[0] + cellradius)
#
#     lowery = max(0, each[1] - cellradius)
#     highery = min(height, each[1] + cellradius)
#
#     for coord in centercoords:
#         xcoord = coord[0]
#         ycoord = coord[1]
#
#         if lowerx <= xcoord and xcoord <= higherx:
#             if lowery <= ycoord and ycoord <= highery:
#                 count += 1
#                 # plt.plot(xcoord, ycoord, 'ro')
#     infectedperdropdict[index] = count
#     index += 1
#
# # plt.savefig('testlabels.jpg')
#
# print("DICTIONARY")
# print(infectedperdropdict)
