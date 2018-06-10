import numpy as np
import cv2
from skimage.morphology import label, reconstruction
import scipy.ndimage as ndimage
import csv
import os
from math import floor
import matplotlib.pyplot as plt
import cv2
import glob

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

def fillhole(img):
    seed = np.copy(img)
    seed[1:-1,1:-1] = np.max(img)
    mask= img
    return reconstruction(seed,mask,method='erosion').astype(np.uint8)

def smooth(img):
    return ndimage.gaussian_filter(img, sigma=(5, 5, 0), order=0)

def newcontrast(im): #enhance the contrast
    if len(im.shape) == 3:
        iml = im[:,:,1]
    else:
        iml = im
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
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

def Intensifier(image):
    return image * 255

#print(preds_test_upsampled[0])
# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

# no cutoff
def centerofrle(rle, height, width):
    toprint = []
    area = height * width
    xrows = []
    ycolumns = []
    for index in range(0, len(rle), 2):
        pixelloc = int(rle[index])
        pixelnum = int(rle[index+1])
        if pixelloc <= area:
            toprint.append(pixelloc)
            for each in range(pixelnum):
                centerpixel = pixelloc + each
                xto = (centerpixel - 1) % height + 1
                xrows.append(xto)
                yto = floor((centerpixel - 1)/height) + 1
                ycolumns.append(yto)
        else:
            break
    midx = sum(xrows)/len(xrows)
    midy = sum(ycolumns)/len(ycolumns)
    return midy, midx

def updateDict(centercoords, cellcoords, bounds):
    infectedlist = []

    for bound in bounds:
        lowery, highery, lowerx, higherx = bound
        count = 0

        for coord in centercoords:
            xcoord = coord[0]
            ycoord = coord[1]

            if lowerx <= xcoord and xcoord <= higherx:
                if lowery <= ycoord and ycoord <= highery:
                    count += 1

        infectedlist.append(count)
    return infectedlist

def truncate(f, n):
  # Truncates/pads a float f to n decimal places without rounding
  s = '{}'.format(str(f))
  if 'e' in s or 'E' in s:
    return '{0:.{1}f}'.format(f, n)
  i, p, d = s.partition('.')
  return '.'.join([i, (d+'0'*n)[:n]])
