# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Utility functions."""

import numpy as np
import time
import tensorflow as tf
from six.moves import xrange

def iou(box1, box2, iouType='segm'):
  """Compute the Intersection-Over-Union of two given boxes.
     or the Intersection-Over box2.
  Args:
    box1: array of 4 elements [cx, cy, width, height].
    box2: same as above
    iouType: The kind of intersection it will compute. 
             'keypoints' is for intersection over box2 area.
  Returns:
    iou: a float number in range [0, 1]. iou of the two boxes.
  """
  lr = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
      max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
  if lr > 0:
    tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
        max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
    if tb > 0:
      intersection = tb*lr
    else:
      intersection = 0
  if(iouType == 'keypoints'):
    box2_area = box2[2] * box2[3]
    return intersection/box2_area
  else:
    union = box1[2]*box1[3]+box2[2]*box2[3]-intersection
    return intersection/union

  return 0

def batch_iou(boxes, box, iouType='segm'):
  """Compute the Intersection-Over-Union of a batch of boxes with another
  box.

  Args:
    box1: 2D array of [cx, cy, width, height].
    box2: a single array of [cx, cy, width, height]
    iouType: the type of Intersection it will compute.
             'keypoints' is for intersection over boxes area.
  Returns:
    ious: array of a float number in range [0, 1].
  """
  lr = np.maximum(
      np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
      np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
      0)
  tb = np.maximum(
      np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
      np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
      0)
  inter = lr*tb
  if(iouType == 'keypoints'):
    boxes_areas = boxes[:,2] * boxes[:,3]
    return inter / boxes_areas
  else:
    union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
    return inter/union

def mass_iou(boxes1, boxes2):
  """
  Args:
    boxes1: 2D array (N,4) of [y_min, x_min, y_max, x_max]
    boxes1: 2D array (M,4) of [y_min, x_min, y_max, x_max]
  """
  xmin1 = boxes1[:, 1]
  ymin1 = boxes1[:, 0]
  xmax1 = boxes1[:, 3]
  ymax1 = boxes1[:, 2]

  xmin2 = boxes2[:, 1]
  ymin2 = boxes2[:, 0]
  xmax2 = boxes2[:, 3]
  ymax2 = boxes2[:, 2]

  if(np.size(xmin1) == 0 or np.size(xmin2) == 0):
    return np.array([]) # avoid computations
  xmax, ymax, xmin, ymin = [], [], [], []
  for i in xrange(np.shape(boxes1)[0]):
    xmax.append(np.minimum(xmax1[i], xmax2))
    ymax.append(np.minimum(ymax1[i], ymax2))
    xmin.append(np.maximum(xmin1[i], xmin2))
    ymin.append(np.maximum(ymin1[i], ymin2))
  xmax = np.array(xmax)
  ymax = np.array(ymax)
  xmin = np.array(xmin)
  ymin = np.array(ymin)
  w = xmax - xmin
  h = ymax - ymin
  intersection = w * h

  w1 = np.tile(np.expand_dims(xmax1 - xmin1, 1), [1, np.size(xmin2)])
  h1 = np.tile(np.expand_dims(ymax1 - ymin1, 1), [1, np.size(xmin2)])
  w2 = np.tile(np.expand_dims(xmax2 - xmin2, 0), [np.size(xmin1), 1])
  h2 = np.tile(np.expand_dims(ymax2 - ymin2, 0), [np.size(xmin1), 1])
  
  union = w1*h2 + w2*h2 - intersection
  
  return intersection / (union + np.finfo(np.float).eps)

def nms(boxes, probs, threshold, iouType='segm'):
  """Non-Maximum supression.
  Args:
    boxes: array of [cx, cy, w, h] (center format)
    probs: array of probabilities
    threshold: two boxes are considered overlapping if their IOU is larger than
        this threshold
    form: 'center' or 'diagonal'
  Returns:
    keep: array of True or False.
  """

  order = probs.argsort()[::-1]
  keep = [True]*len(order)

  for i in range(len(order)-1):
    ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]], iouType)
    for j, ov in enumerate(ovps):
      if ov > threshold:
        keep[order[j+i+1]] = False
  return keep

# TODO(bichen): this is not equivalent with full NMS. Need to improve it.
def recursive_nms(boxes, probs, threshold, form='center'):
  """Recursive Non-Maximum supression.
  Args:
    boxes: array of [cx, cy, w, h] (center format) or [xmin, ymin, xmax, ymax]
    probs: array of probabilities
    threshold: two boxes are considered overlapping if their IOU is largher than
        this threshold
    form: 'center' or 'diagonal'
  Returns:
    keep: array of True or False.
  """

  assert form == 'center' or form == 'diagonal', \
      'bounding box format not accepted: {}.'.format(form)

  if form == 'center':
    # convert to diagonal format
    boxes = np.array([bbox_transform(b) for b in boxes])

  areas = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
  hidx = boxes[:, 0].argsort()
  keep = [True]*len(hidx)

  def _nms(hidx):
    order = probs[hidx].argsort()[::-1]

    for idx in xrange(len(order)):
      if not keep[hidx[order[idx]]]:
        continue
      xx2 = boxes[hidx[order[idx]], 2]
      for jdx in xrange(idx+1, len(order)):
        if not keep[hidx[order[jdx]]]:
          continue
        xx1 = boxes[hidx[order[jdx]], 0]
        if xx2 < xx1:
          break
        w = xx2 - xx1
        yy1 = max(boxes[hidx[order[idx]], 1], boxes[hidx[order[jdx]], 1])
        yy2 = min(boxes[hidx[order[idx]], 3], boxes[hidx[order[jdx]], 3])
        if yy2 <= yy1:
          continue
        h = yy2-yy1
        inter = w*h
        iou = inter/(areas[hidx[order[idx]]]+areas[hidx[order[jdx]]]-inter)
        if iou > threshold:
          keep[hidx[order[jdx]]] = False

  def _recur(hidx):
    if len(hidx) <= 20:
      _nms(hidx)
    else:
      mid = len(hidx)/2
      _recur(hidx[:mid])
      _recur(hidx[mid:])
      _nms([idx for idx in hidx if keep[idx]])

  _recur(hidx)

  return keep

def sparse_to_dense(sp_indices, output_shape, values, default_value=0):
  """Build a dense matrix from sparse representations.

  Args:
    sp_indices: A [0-2]-D array that contains the index to place values.
    shape: shape of the dense matrix.
    values: A {0,1}-D array where values corresponds to the index in each row of
    sp_indices.
    default_value: values to set for indices not specified in sp_indices.
  Return:
    A dense numpy N-D array with shape output_shape.
  """

  assert len(sp_indices) == len(values), \
      'Length of sp_indices is not equal to length of values'

  array = np.ones(output_shape) * default_value
  for idx, value in zip(sp_indices, values):
    array[tuple(idx)] = value
  return array

def bgr_to_rgb(ims):
  """Convert a list of images from BGR format to RGB format."""
  out = []
  for im in ims:
    out.append(im[:,:,::-1])
  return out

def bbox_transform(bbox):
  """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
  for numpy array or list of tensors.
  """
  with tf.variable_scope('bbox_transform') as scope:
    cx, cy, w, h = bbox
    out_box = [[]]*4
    out_box[0] = cx-w/2
    out_box[1] = cy-h/2
    out_box[2] = cx+w/2
    out_box[3] = cy+h/2

  return out_box

def bbox_transform_inv(bbox):
  """convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]. Works
  for numpy array or list of tensors.
  """
  with tf.variable_scope('bbox_transform_inv') as scope:
    xmin, ymin, xmax, ymax = bbox
    out_box = [[]]*4

    width       = xmax - xmin + 1.0
    height      = ymax - ymin + 1.0
    out_box[0]  = xmin + 0.5*width 
    out_box[1]  = ymin + 0.5*height
    out_box[2]  = width
    out_box[3]  = height

  return out_box

def safe_exp(w, thresh):
  """Safe exponential function for tensors."""

  slope = np.exp(thresh)
  with tf.variable_scope('safe_exponential'):
    lin_bool = w > thresh
    lin_region = tf.to_float(lin_bool)

    lin_out = slope*(w - thresh + 1.)
    exp_out = tf.exp(tf.where(lin_bool, tf.zeros_like(w), w))

    out = lin_region*lin_out + (1.-lin_region)*exp_out
    # TODO: all above 3 lines can be calculated with the line below
    # out = np.where(lin_bool, lin_out, np.exp(w))
  return out