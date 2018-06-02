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

import tensorflow as tf

is_training = False
image_class = 0

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage3_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

trainlist = []

def fillhole(img):
    seed = np.copy(img)
    seed[1:-1,1:-1] = np.max(img)
    mask= img
    return reconstruction(seed,mask,method='erosion').astype(np.uint8)

def smooth(img):
    return ndimage.gaussian_filter(img, sigma=(5, 5, 0), order=0)

def newcontrast(im): #enhance the contrast
#    print(im.shape)
#    print(im)
    if len(im.shape) == 3:
        iml = im[:,:,1]
    else:
        iml = im
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
#    print(iml.shape)
#    print(iml)
    iml = clahe.apply(iml)
    if len(im.shape) == 3:
        for i in range(3): im[:,:,i] == iml
    else:
        im = iml
    return imfloat(im)

def imfloat(im):
    return im #img_as_float(im) # return the image in (0,1)

def recontrast(im):
    return im

def fcontrast(im):
#    return imfloat(im)
    return newcontrast(im)

def imclass(im):  # black-white image check
    aim = im
    if len(im.shape) == 3: # is it RGB?
        aim = im[:,:,1] # green layer selected
    imhist = np.histogram(aim, bins=list(range(256)))
    if np.argmax(imhist[0]) < 82:  # regular black background
#    if np.sum(imhist[0][0:82]) > np.sum(imhist[0][173:255]):
        return 0, recontrast(im)
    else:    # invert the image because of white background
        aim = np.invert(aim)
        if len(im.shape) == 3:
            for i in range(3): im[:,:,i] = aim
        else:
            im = aim
        return 1, recontrast(im)

def imscale(im):  # gray scale or histologic?
    simg = np.sum(im[:,:,0]-im[:,:,1])
    if simg == 0: return imclass(im)   # gray scale image confirmed
    base = np.invert(im[:,:,1])   # invert histologic image scale
    for i in range(3):
        im[:,:,i] = base
    return 2, recontrast(im) #enhance the contrast

def x_ext(i,N):
    if i > -1 and i < N:
        return i
    elif i < 2*N:
        return 2*N -1-i
    else:
        return x_ext(i-2*N,N)

def img_ext(im,lb):
    N, M, _ = im.shape
    for xs in range(IMG_WIDTH):
        for ys in range(IMG_HEIGHT):
            img = np.zeros((IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS),dtype=np.uint8)
            label = np.zeros((IMG_WIDTH,IMG_HEIGHT,1),dtype=np.bool)
            for i in range(xs,xs+IMG_WIDTH):
                for j in range(ys,ys+IMG_HEIGHT):
                    img[i,j,:] = im[x_ext(i,N),x_ext(j,M),:]
                    label[i,j,0] = lb[x_ext(i,N),x_ext(j,M)]
            if np.sum(label) != 0:
                return img, label

def img_resampled(im,lb):
    N, M, _ = im.shape
    if N == IMG_WIDTH and M == IMG_HEIGHT:
        return [im], [np.expand_dims(lb, axis=-1)]

    nn = (N-1) // IMG_WIDTH
    mm = (M-1) // IMG_HEIGHT

    imgs, labels = [], []
    for i in range(nn+1):
        for j in range(mm+1):
            img = np.zeros((IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS),dtype=np.uint8)
            label = np.zeros((IMG_WIDTH,IMG_HEIGHT,1),dtype=np.bool)
            for x in range(IMG_WIDTH):
                for y in range(IMG_HEIGHT):
                    sx = i*IMG_WIDTH + x
                    sy = j*IMG_HEIGHT + y
                    t = im[x_ext(sx,N),x_ext(sy,M),1]
                    img[x,y,:] = np.array([t,t,t])
                    label[x,y,0] = lb[x_ext(sx,N),x_ext(sy,M)]
            if np.sum(label) != 0:
                imgs.append(img)
                labels.append(label)
    return imgs, labels

def img_sample(im):
    N, M, _ = im.shape
    if N == IMG_WIDTH and M == IMG_HEIGHT: return [im]
    nn = (N-1) // (IMG_WIDTH // 2)
    mm = (M-1) // (IMG_HEIGHT // 2)
    imgs = []
    for i in range(nn+1):
        for j in range(mm+1):
            img = np.zeros((IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS),dtype=np.uint8)
            for x in range(IMG_WIDTH):
                for y in range(IMG_HEIGHT):
                    sx = i*IMG_WIDTH // 2 + x
                    sy = j*IMG_HEIGHT // 2 + y
                    t = im[x_ext(sx,N),x_ext(sy,M),1]
                    img[x,y,:] = np.array([t,t,t])
            imgs.append(img)
    return imgs

def img_assembly(imgs,shape):

    N, M, _ = shape
    if N == IMG_WIDTH and M == IMG_HEIGHT: return imgs[0]

    nn = (N-1) // (IMG_WIDTH // 2)
    mm = (M-1) // (IMG_HEIGHT // 2)
    im = np.zeros((N,M,IMG_CHANNELS),dtype=np.uint8)
    for i in range(nn+1):
        for j in range(mm+1):
            k = i*(mm+1) + j
            for x in range(IMG_WIDTH):
                for y in range(IMG_HEIGHT):
                    sx = i*IMG_WIDTH // 2 + x
                    sy = j*IMG_HEIGHT // 2 + y
                    if sx < N and sy < M:
                        im[sx,sy,:] = np.maximum((imgs[k])[x,y,:], im[sx,sy,:])
    return im

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
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    bw, img = imscale(img)
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
        imgs, masks = img_resampled(img, mask)
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
    X_train[i] = fcontrast(XX_train[i])
    Y_train[i] = YY_train[i]

print('\n Read ',len(trainlist),'(%d)'%m,' samples of class',image_class,',',len(train_ids),' left')

del XX_train, YY_train
gc.collect()

# Check if training data looks all right
# ix = random.randint(0, len(X_train))
# imshow(X_train[ix])
# plt.show()
# imshow(np.squeeze(Y_train[ix]))
# plt.show()

# Get and resize test images
XX_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Xtest = []
sizes_test = []
shape_test = []
scale_test = [0]
non_ids = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    bw, img = imscale(img)
    if bw == image_class:
        sizes_test.append([img.shape[0], img.shape[1]])
        shape_test.append([img.shape])
    else:
        non_ids.append(id_)
#    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    imgs = img_sample(img)
    if bw == image_class:
        scale_test.append(scale_test[-1]+len(imgs))
        for img in imgs:
            Xtest.append(img)
#        XX_test[m] = img

X_test = np.zeros((len(Xtest), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
for i, ximg in enumerate(Xtest):
    X_test[i] = fcontrast(ximg)

del XX_test
gc.collect()

for untest in non_ids:
    test_ids.remove(untest)

print('Read ',len(sizes_test), 'samples of class',image_class,',',len(test_ids),' left')

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

if is_training:
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    model.summary()
    # Fit model
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model-dsbowl.h5', verbose=1, save_best_only=True)
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,
                        callbacks=[earlystopper, checkpointer])
else:
# load json and create model
    print("reading previously saved model")
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

print("predicting based on the model")
# Predict on train, val and test
if is_training:
    preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
    preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Smoothing
#for i, img in enumerate(preds_test):
#    preds_test[i] = smooth(img)

# Threshold predictions
if is_training:
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# convert resampled test cases to original scale
preds_test_a = []
for i in range(len(shape_test)):
    img = img_assembly(preds_test_t[scale_test[i]:scale_test[i+1]],shape_test[i][0])
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

def Intensifier(image):
    return image * 255

#for i,cc in enumerate(preds_test_upsampled):
##    cc = cc*255
###    cc = fillhole(cc) # not much help
##    cc = smooth(cc)
##    cc = (cc > 128).astype(np.uint8)
###    preds_test_upsampled[i] = cc
#    cv2.imwrite('predmap'+str(i)+'.png',cc*255)

# show predicted samples
for i in range(len(preds_test_a)):
    print(i,preds_test_a[i].shape,np.min(np.squeeze(preds_test_a[i])),np.mean(np.squeeze(preds_test_a[i])) )
    imshow(np.squeeze(Intensifier(preds_test_upsampled[i])),cmap='gray')
    plt.show()

#print(preds_test_upsampled[0])
# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

if is_training:
    # serialize model to JSON
    model_json = model.to_json()
    with open("mmodel"+str(image_class)+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("mmodel"+str(image_class)+".h5")
    print("Saved model to disk")

with open('bwtrain'+str(image_class)+'.log', 'w') as myfile:
    wr = csv.writer(myfile)
    for bw in trainlist:
        wr.writerow([bw])

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl'+str(image_class)+'.csv', index=False)
