import tensorflow as tf
import numpy as np
import sys
import os

### TENSORFLOW SETUP
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("logs/trained_labels.txt")]

with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  g2 = tf.import_graph_def(graph_def, name='g2')

def isball(image_data):
  with tf.Session(graph=g2) as sess2:
    softmax_tensor = sess2.graph.get_tensor_by_name('g2/final_result:0')
    predictions = sess2.run(softmax_tensor, {'g2/DecodeJpeg:0': image_data})
    return predictions[0]
