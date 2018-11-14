import numpy as np
import tensorflow as tf
from tensorflow.contrib.image.python.ops import image_ops
from utils.tf_util import batch_iou, batch_bbox_transform, compute_distances


def get_matcher_algorithm(matcher_name):
  """
  Args:
    mather_name: name of the matcher algorithm
  Returns:
    A function object implementing the algorithm. This function object accepts arguments:
    record: The record is a dictionary of tensors after the data input stage
    anchor_box_tens: Tensor of anchor boxes represented in the form [xmin,ymin,xmax,ymax]
    dataset_box_tens: Tensor of ground truth boxes, read from the dataset, in the form [xmin,ymin,xmax,ymax]
    anchor_box_centers: Tensor of anchor boxes represented in the form [x,y,w,h]
    dataset_box_centers: Tensor of ground truth boxes, read from the dataset, in the form [x,y,w,h]
    idx: The indices of the returning sparse tensor.
    N: number of boxes in anchor_box_tens (equal to number of boxes in anchor_box_centers).
    M: number of boxes in dataset_box_tens (equal to number of boxes in dataset_box_centers).
    pre_N: numpy number of N (not a tensorflow tensor).
    batch_size: The batch size.
    The function returns a sparse tensor with values the indices of 
    the matching anchors of the ground truth boxes inside the anchor_box_tens tensor.
  """
  matchers  = {
    "GREEDY_PARALLELIZABLE_MATCH" : greedy_parallelizable_match,
    "GREEDY_BIPARTITE_MATCH"      : greedy_bipartite_match
  }
  if matcher_name in matchers.keys():
    return matchers[matcher_name]
  else:
    print("No valid matcher selected, selecting default matcher: GREEDY_PARALLELIZABLE_MATCH")
  
  return matchers["GREEDY_PARALLELIZABLE_MATCH"]

def greedy_parallelizable_match(record, anchor_box_tens, dataset_box_tens, anchor_box_centers, dataset_box_centers, idx, N, M, pre_N, batch_size):
  """
  This function uses the algorithm described in the coresponding paper.
  It tries to solve the bipartite match by taking each dataset box first and then picks the closest anchor and assigns it as
  responsible for this dataset box. It also takes care that the same anchor is not responsible for more than one dataset box.
  Args:
    Described in get_matcher_algorithm
  Returns:
    Described in get_matcher_algorithm
  """
  flat_overlaps = batch_iou(anchor_box_tens,
                            dataset_box_tens,
                            N=N, M=M) # [SIZE_XMIN, N] = [M, N]
  
  # sort for biggest overlaps 
  max_overlaps = tf.reduce_max(flat_overlaps, axis=1) #[SIZE_XMIN]
  overlaps_sorted_idx = tf.contrib.framework.argsort(flat_overlaps, axis=1, direction='DESCENDING') # [SIZE_XMIN, N]

  # get distances of between whole boxes, d([x1,y1,w1,h1], [x2,y2,w2,h2])
  flat_dists = compute_distances(anchor_box_centers,
                                  dataset_box_centers,
                                  N=N, M=M) # [SIZE_XMIN, N] = [M, N]

  # sort for minimum dists
  dists_sorted_idx = tf.contrib.framework.argsort(flat_dists, axis=1, direction='ASCENDING') # [SIZE_XMIN, N]

  aidx_array = tf.where(tf.reshape(
                          tf.tile(tf.expand_dims(max_overlaps, axis=1), [1, pre_N]), tf.shape(overlaps_sorted_idx)) <= 0,
                        dists_sorted_idx,
                        overlaps_sorted_idx) # [SIZE_XMIN, N]

  unstacked_aidx_array = tf.dynamic_partition(aidx_array, tf.cast(idx[:,0], dtype=tf.int32), batch_size, name="dynamic_partition")

  aidx_values = tf.concat([find_best_aidx_per_image(el) for el in unstacked_aidx_array] , axis=0)

  aidx = tf.SparseTensor(idx, aidx_values, record["image/object/bbox/xmin"].dense_shape)

  return aidx

def greedy_bipartite_match(record, anchor_box_tens, dataset_box_tens, anchor_box_centers, dataset_box_centers, idx, N, M, pre_N, batch_size):
  """
  This algorithm described as greed bipartite match.
  It iterates through all the dataset boxes.
  At each iteration, it picks the closest distance between an achor box and a dataset box.
  It also takes care that the same anchor is not responsible for more than one dataset box.
  Args:
    Described in get_matcher_algorithm
  Returns:
    Described in get_matcher_algorithm
  """
  flat_overlaps = batch_iou(anchor_box_tens,
                            dataset_box_tens,
                            N=N, M=M) # [SIZE_XMIN, N] = [M, N]

  max_overlaps = tf.reduce_max(flat_overlaps, axis=1) #[SIZE_XMIN]

  # get distances of between whole boxes, d([x1,y1,w1,h1], [x2,y2,w2,h2])
  flat_dists = compute_distances(anchor_box_centers,
                                  dataset_box_centers,
                                  N=N, M=M) # [SIZE_XMIN, N] = [M, N]
  # get a combined distance matrix
  dst_mtxs = tf.where(tf.reshape(
                        tf.tile(tf.expand_dims(max_overlaps, axis=1), [1, pre_N]), tf.shape(flat_overlaps)) <= 0,
                      flat_dists,
                      1.0 - flat_overlaps)
  
  unstacked_dst_mtxs = tf.dynamic_partition(dst_mtxs, tf.cast(idx[:,0], dtype=tf.int32), batch_size, name="dynamic_partition")
  
  aidx_values = tf.concat([image_ops.bipartite_match(el, num_valid_rows=-1.0)[0] for el in unstacked_dst_mtxs] , axis=0)

  aidx = tf.SparseTensor(idx, aidx_values, record["image/object/bbox/xmin"].dense_shape)

  return aidx


def find_best_aidx_per_image(aidx_slice):
  """
    This function is the used for a greedy parallelizable matching of anchor boxes to dataset boxes.
    It accepts an aidx_slice argument which is a 2D tensor (matrix) for every image, with every row containing 
    the indices of anchors sorted using the distance of the dataset box regarding this row with the 
    anchor boxes as a key.
    Args:
      aidx_slice: a matrix with anchor indices, each row corresponds to a dataset box.
    Returns:
      A greedy match for each dataset box as a 1D Tensor.
  """
  with tf.name_scope("aidx_elimination"):
    els_used = tf.zeros(tf.shape(aidx_slice)[:1]-1, dtype=tf.int32) # keep all used elements so far
    i = tf.constant(1, dtype=tf.int32)
    els_used = tf.reshape(tf.concat([tf.expand_dims(aidx_slice[0, 0], 0), els_used], axis=0), tf.shape(aidx_slice)[:1])

    neg_ones = -tf.ones_like(aidx_slice) #  mark with -1 elements of N that have already been used
    N = aidx_slice
    c = lambda i, N, els_used : i < tf.shape(N)[0]
    b = lambda i, N, els_used : [i+1, tf.where(tf.equal(N, els_used[i-1]), neg_ones, N), 
                                els_used + 
                                  tf.sparse_to_dense(i, tf.shape(N)[:1], 
                                                    tf.gather(N[i, :], 
                                                      tf.reduce_min(
                                                        tf.where(
                                                          tf.logical_and(
                                                            tf.not_equal(N[i, :], -1), 
                                                            tf.not_equal(N[i,:], els_used[i-1]))))))]
    return tf.while_loop(c, b, [i, N, els_used])[2]