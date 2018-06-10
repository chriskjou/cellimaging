import tensorflow as tf
import numpy as np
import sys
import os
import cv2

### TENSORFLOW SETUP
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("logs-fluorescence/trained_labels.txt")]

with tf.gfile.FastGFile("logs-fluorescence/trained_graph.pb", 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  g2 = tf.import_graph_def(graph_def, name='g2')

def howmany(preds):
    maxindex = 0
    for i in range(len(preds)):
        if preds[i] > preds[maxindex]:
            maxindex = i

    # label order in predictions
    labels = [8, 5, 4, 1, 7, 6, 3, 2, 0]
    return labels[maxindex]

def cellnum(image_data):
  with tf.Session(graph=g2) as sess2:
    softmax_tensor = sess2.graph.get_tensor_by_name('g2/final_result:0')
    predictions = sess2.run(softmax_tensor, {'g2/DecodeJpeg:0': image_data})

    return howmany(predictions[0])

def countinfectedcells(img, bounds):
    image = cv2.imread(img)
    counts = []
    for bound in bounds:
        lowery, highery, lowerx, higherx = bound
        counts.append(cellnum(image[lowery: highery, lowerx: higherx]))
    return counts
