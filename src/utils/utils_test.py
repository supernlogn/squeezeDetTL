import numpy as np
import tensorflow as tf

import util
import tf_util




def test_batch_iou_vs_massive_iou(sess = None):
  WIDTH = 1248.0
  HEIGHT = 384.0
  N = 3000
  M = 200
  xmin1 = np.random.random([N]) * WIDTH
  ymin1 = np.random.random([N]) * HEIGHT
  boxes1 = {
    "xmin" : xmin1,
    "xmax" : xmin1 + np.random.random([N]) * WIDTH,
    "ymin" : ymin1,
    "ymax" : ymin1 + np.random.random([N]) * HEIGHT,
  }
  boxes1_array = np.stack([boxes1["ymin"], boxes1["xmin"], boxes1["ymax"], boxes1["xmax"]], axis=-1)
  boxes1_tens = {tens_name : tf.constant(boxes1[tens_name], dtype=tf.float32)
                       for tens_name in boxes1.keys()}
  xmin2 = np.random.random([M]) * WIDTH
  ymin2 = np.random.random([M]) * HEIGHT
  boxes2 = {
    "xmin" : xmin2,
    "xmax" : xmin2 + np.random.random([M]) * WIDTH,
    "ymin" : ymin2,
    "ymax" : ymin2 + np.random.random([M]) * HEIGHT,
  }
  boxes2_array = np.stack([boxes2["ymin"], boxes2["xmin"], boxes2["ymax"], boxes2["xmax"]], axis=-1)
  boxes2_tens = {tens_name : tf.constant(boxes2[tens_name], dtype=tf.float32)
                       for tens_name in boxes1.keys()}
  tens = tf_util.batch_iou(boxes1_tens, boxes2_tens, tf.constant(N), tf.constant(M))
  
  if not sess:
    sess = tf.Session()
  # computed
  c_batch_iou = sess.run(tens)
  # ground truth
  boxes1_centers = np.array([util.bbox_transform_inv(box1) for box1 in boxes1_array]).astype(np.float32)
  boxes2_centers = np.array([util.bbox_transform_inv(box2) for box2 in boxes2_array]).astype(np.float32)
  # g_batch_iou = util.mass_iou(boxes2_array, boxes1_array)
  g_batch_iou = []
  for i in range(len(xmin2)):
    overlaps = util.batch_iou(boxes1_centers, boxes2_centers[i])
    # for j in range(len(xmin1)):
    #   overlaps.append(util.iou(boxes1_centers[j], boxes2_centers[i]))
    g_batch_iou.append(overlaps)
  g_batch_iou = np.array(g_batch_iou)
  print(np.shape(c_batch_iou))
  print(np.shape(g_batch_iou))
  # print(c_batch_iou)
  print(np.abs(c_batch_iou - g_batch_iou))
  print(np.max(np.abs(np.argsort(c_batch_iou, axis=1) - np.argsort(g_batch_iou, axis=1))))
  print(np.max(np.abs(c_batch_iou- g_batch_iou)))

def test_compute_distances(sess=None):
  N = 10
  M = 20
  box_centers1 = {
    "x" : np.random.random([N]),
    "y" : np.random.random([N])
  }
  box_centers1_tens = {
    "x" : tf.constant(box_centers1["x"]),
    "y" : tf.constant(box_centers1["y"])
  }
  box_centers2 = {
    "x" : np.random.random([M]),
    "y" : np.random.random([M])
  }
  box_centers2_tens = {
    "x" : tf.constant(box_centers2["x"]),
    "y" : tf.constant(box_centers2["y"])
  }
  if not sess:
    sess = tf.Session()
  # computed
  c_dists2 = sess.run(tf_util.compute_distances(box_centers2_tens, box_centers1_tens, tf.constant(M), tf.constant(N))) # [N,M]
  # ground truth
  dX = np.subtract.outer(box_centers1["x"], box_centers2["x"]) # [N, M]
  dY = np.subtract.outer(box_centers1["y"], box_centers2["y"]) # [N, M]
  g_dists2 = dX * dX + dY * dY

  print(np.max(np.abs(c_dists2 - g_dists2)))

if __name__ == "__main__":
  test_batch_iou_vs_massive_iou()
  # test_compute_distances()
  