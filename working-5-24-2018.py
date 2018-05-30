#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:56:02 2018

Version 3: identify type of images, change contraast,
           and train each type differently
Version 4: eliminate image scaling
        4a: train with clipped images
        4b: train with overlapped images

@author: chriskjou
"""

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

is_training = False
image_class = 0

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage2_test/'
testfiles = glob.glob("stage3_test/*.png")

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

trainlist = []

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

non_ids = [] # store samples that are not in class needed
# Get and resize train images and masks
XX_train = []
YY_train = []
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
m = 0
if is_training:
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        bw, img = H.imscale(img)
    #    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
    #        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
    #                                      preserve_range=True), axis=-1)
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

# Get and resize test images
XX_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Xtest = []
sizes_test = []
shape_test = []
scale_test = [0]
non_ids = []
print('Getting and resizing test images ... ')
sys.stdout.flush()

for each in testfiles:
    img = imread(each)[:,:,:IMG_CHANNELS]
    # RESIZE FOR IMAGES
    # ALLOW FOR NON PNGS
    bw, img = H.imscale(img)
    if bw == image_class:
        sizes_test.append([img.shape[0], img.shape[1]])
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

del XX_test
gc.collect()

for untest in non_ids:
    test_ids.remove(untest)

print('Read ',len(sizes_test), 'samples of class',image_class,',',len(test_ids),' left')

if is_training:
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[H.mean_iou])
    model.summary()
    # Fit model
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model-dsbowl.h5', verbose=1, save_best_only=True)
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,
                        callbacks=[earlystopper, checkpointer])
else:
# load json and create model
    print("reading previously saved model...\n")
    json_file = open('mmodel'+str(image_class)+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("mmodel"+str(image_class)+".h5")
#    loaded_model = model_from_json(loaded_model_json)
#    # load weights into new model
#    loaded_model.load_weights("model.h5")
#    print("Loaded model from disk")
#    model = load_model('model-dsbowl.h5', custom_objects={'mean_iou': mean_iou})

print("predicting based on the model...\n")
# Predict on train, val and test
if is_training:
    preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
    preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Smoothing
#for i, img in enumerate(preds_test):
#    preds_test[i] = H.smooth(img)

# Threshold predictions
if is_training:
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# convert resampled test cases to original scale
preds_test_a = []
for i in range(len(shape_test)):
    img = H.img_assembly(preds_test_t[scale_test[i]:scale_test[i+1]],shape_test[i][0])
    preds_test_a.append(img)

# Create list of upsampled test masks
#preds_test_upsampled = []
#for i in range(len(preds_test)):
#    preds_test_upsampled.append(resize(np.squeeze(preds_test_t[i]),
#                                       (sizes_test[i][0], sizes_test[i][1]),
#                                       mode='constant', preserve_range=True))

# convert resampled test cases to original scale
preds_test_upsampled = []
for i in range(len(preds_test_a)):
    preds_test_upsampled.append(preds_test_a[i])

if is_training:
    # Perform a sanity check on some random training samples
    ix = random.randint(0, len(preds_train_t))
    imshow(X_train[ix],cmap='gray')
    plt.show()
    imshow(np.squeeze(Y_train[ix]),cmap='gray')
    plt.show()
    imshow(np.squeeze(preds_train_t[ix]),cmap='gray')
    plt.show()

    # Perform a sanity check on some random validation samples
    ix = random.randint(0, len(preds_val_t))
    imshow(X_train[int(X_train.shape[0]*0.9):][ix],cmap='gray')
    plt.show()
    imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]),cmap='gray')
    plt.show()
    imshow(np.squeeze(preds_val_t[ix]))
    #imshow(np.squeeze(preds_val[ix]),cmap='gray')
    plt.show()

#for i,cc in enumerate(preds_test_upsampled):
##    cc = cc*255
###    cc = H.fillhole(cc) # not much help
##    cc = H.smooth(cc)
##    cc = (cc > 128).astype(np.uint8)
###    preds_test_upsampled[i] = cc
#    cv2.imwrite('predmap'+str(i)+'.png',cc*255)

# show predicted samples
# for i in range(len(preds_test_a)):
#     print(i,preds_test_a[i].shape,np.min(np.squeeze(preds_test_a[i])),np.mean(np.squeeze(preds_test_a[i])) )
#     imshow(np.squeeze(H.Intensifier(preds_test_upsampled[i])),cmap='gray')
#     plt.show()

new_test_ids = []
rles = []

### ADAPTED ###
index = 0
for each in testfiles:
    rle = list(H.prob_to_rles(preds_test_upsampled[index]))
    rles.extend(rle)
    new_test_ids.extend([each] * len(rle))
    index += 1

if is_training:
    # serialize model to JSON
    model_json = model.to_json()
    with open("mmodel"+str(image_class)+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("mmodel"+str(image_class)+".h5")
    print("Saved model to disk...")

with open('bwtrain'+str(image_class)+'.log', 'w') as myfile:
    wr = csv.writer(myfile)
    for bw in trainlist:
        wr.writerow([bw])

# Create submission DataFrame
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
# sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

actual.to_csv('cellcount'+str(image_class)+'.csv', index=False)
print("Done adding cell counts to CSV.")
