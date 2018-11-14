# Author: Ioannis Athanasiadis (ath.ioannis94@gmail.com) 04/25/2018

"""Utility functions."""

import time

import numpy as np

import tensorflow as tf


def batch_iou(boxes1, boxes2, N, M, EPSILON=np.finfo(np.float32).eps):
  """ Computes the iou between two arrays
      of boxes.
      Args:
        boxes1: a python dictionary containing
                {
                  "xmin" : Tensor of shape [N]
                  "xmax" : Tensor of shape [N]
                  "ymin" : Tensor of shape [N]
                  "ymax" : Tensor of shape [N]
                }
        boxes2: a python dictionary containing
                {
                  "xmin" : Tensor of shape [M]
                  "xmax" : Tensor of shape [M]
                  "ymin" : Tensor of shape [M]
                  "ymax" : Tensor of shape [M]
                }
        N, M : 0-D Tensors of M,N sizes
      Returns:
        Intersection over union with shape [M,N]
  """
  # TODO: look util.mass_iou to avoid transpose and reshape
  with tf.name_scope('intersection'):
    xmin1 = boxes1["xmin"]
    xmin2 = boxes2["xmin"]
    xmin = tf.maximum(xmin1, tf.expand_dims(xmin2, 1), name='xmin')

    xmax1 = boxes1["xmax"] + 1.0
    xmax2 = boxes2["xmax"] + 1.0
    xmax = tf.minimum(xmax1, tf.expand_dims(xmax2, 1), name='xmax')
    w = tf.maximum(tf.constant(0, dtype=xmax.dtype), xmax-xmin, name='inter_w')

    ymin1 = boxes1["ymin"]
    ymin2 = boxes2["ymin"]
    ymin = tf.maximum(ymin1, tf.expand_dims(ymin2, 1), name='ymin')

    ymax1 = boxes1["ymax"] + 1.0
    ymax2 = boxes2["ymax"] + 1.0
    ymax = tf.minimum(ymax1, tf.expand_dims(ymax2, 1), name='ymax')
    h = tf.maximum(tf.constant(0, dtype=ymax.dtype), ymax-ymin, name='inter_h')

    intersection = tf.multiply(w, h, name='intersection')

  with tf.variable_scope('union'):
    w1 = tf.subtract(xmax1, xmin1, name='w1')
    h1 = tf.subtract(ymax1, ymin1, name='h1')
    w2 = tf.subtract(xmax2, xmin2, name='w2')
    h2 = tf.subtract(ymax2, ymin2, name='h2')

    union = tf.cast(tf.expand_dims(w1*h1,0) + tf.expand_dims(w2*h2, 1) - intersection, dtype=tf.float32)

  return tf.truediv(tf.cast(intersection, dtype=tf.float32), union)

def compute_distances(box_centers1, box_centers2, N, M):
  """
    Args:
      box_centers1 : dict of {
        "x" : tensor array of center's x of shape [N]
        "y" : tensor array of centers' y of shape [N]
        "w" : tensor array of centers' w of shape [N]
        "h" : tensor array of centers' h of shape [N]
      }
      box_centers2 : dict of {
        "x" : tensor array of center's x of shape [M]
        "y" : tensor array of centers' y of shape [M]
        "w" : tensor array of centers' w of shape [M]
        "h" : tensor array of centers' h of shape [M]
      }
    Returns:
      a Tensor with shape [M,N] where all the squared distances 
      between bbox_centers1 and box_centers2 are computed.
  """
  with tf.variable_scope("box_distances"):
    dx2 = tf.square(tf.expand_dims(box_centers2["x"], 1) - box_centers1["x"])
    dy2 = tf.square(tf.expand_dims(box_centers2["y"], 1) - box_centers1["y"])  
    dw2 = tf.square(tf.expand_dims(box_centers2["w"], 1) - box_centers1["w"])  
    dh2 = tf.square(tf.expand_dims(box_centers2["h"], 1) - box_centers1["h"])
    sq_dist = dx2 + dy2 + dw2 + dh2 
  
  return sq_dist

def batch_bbox_transform(bbox):
  """convert a batch of bboxes of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
     for numpy array or list of tensors.
  """
  reshaped_box = np.reshape(bbox, newshape = (-1, 4))

  xmin = reshaped_box[:,0] - reshaped_box[:,2]/2
  ymin = reshaped_box[:,1] - reshaped_box[:,3]/2
  xmax = reshaped_box[:,0] + reshaped_box[:,2]/2
  ymax = reshaped_box[:,1] + reshaped_box[:,3]/2
  return {
      "xmin" : xmin,
      "ymin" : ymin,
      "xmax" : xmax,
      "ymax" : ymax}

def _tf_dense_to_sparse_tensor(idx, tens):
  """
    Returns:
      a sparse tensor by taking the elements 
      of tens described in inidices.
  """
  return tf.SparseTensor(idx, tf.gather_nd(tens, idx), \
                         tf.shape(tens, out_type=tf.int64))
