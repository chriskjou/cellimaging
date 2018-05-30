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

image_class = 0

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage2_test/'
testfiles = glob.glob("005/*.tiff")

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

# Get and resize test images
XX_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Xtest = []
sizes_test = []
shape_test = []
scale_test = [0]
non_ids = []
print('Getting and resizing test images ... ')
sys.stdout.flush()

print(testfiles)
for each in testfiles:
    print(each)
    # if tiff
    imgpath = each.split(".")
    imgtype = imgpath[-1]
    grayscale = Image.open(each).convert('L')
    grayscale.save('temp.png', "PNG")
    png = cv2.imread('temp.png')
    gspng = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
    torgb = cv2.cvtColor(gspng, cv2.COLOR_GRAY2BGR)
    img = torgb[:,:,:IMG_CHANNELS]

    # if png
    img = cv2.imread(each)[:,:,:IMG_CHANNELS]

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

print('Read ',len(sizes_test), 'samples of class',image_class,',',len(test_ids),' left')

# load json and create model
print("reading previously saved model...\n")
json_file = open('mmodel'+str(image_class)+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("mmodel"+str(image_class)+".h5")

print("predicting based on the model...\n")

preds_test_t = (preds_test > 0.5).astype(np.uint8)

# convert resampled test cases to original scale
preds_test_a = []
for i in range(len(shape_test)):
    img = H.img_assembly(preds_test_t[scale_test[i]:scale_test[i+1]],shape_test[i][0])
    preds_test_a.append(img)

# convert resampled test cases to original scale
preds_test_upsampled = []
for i in range(len(preds_test_a)):
    preds_test_upsampled.append(preds_test_a[i])

new_test_ids = []
rles = []

### ADAPTED ###
index = 0
for each in testfiles:
    rle = list(H.prob_to_rles(preds_test_upsampled[index]))
    rles.extend(rle)
    new_test_ids.extend([each] * len(rle))
    index += 1

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
