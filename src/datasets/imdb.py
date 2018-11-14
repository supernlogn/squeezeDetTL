import os
import sys
import json
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
from sklearn.cluster import KMeans
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from easydict import EasyDict as edict
from config import config_cooker
from utils.tf_util import batch_iou, batch_bbox_transform, compute_distances, _tf_dense_to_sparse_tensor
import utils.util as util
import creation.dataset_util as dataset_util
from matchers import get_matcher_algorithm

def get_keys_to_features():
  keys_to_features = {
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/object/bbox/xmin": tf.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.VarLenFeature(tf.float32),
        "image/object/class/label": tf.VarLenFeature(tf.int64)
  }
  return keys_to_features

def load_data(record, mc, training=True, image_decoder=tf.image.decode_png):
  """
    In this function all preprocessing of the dataset data prior to the feature extractor is performed. Here both data augmentation
    and 
    Args:
      record: dictionary with input tensors see `get_keys_to_features` to see the keys of this dictionary
      mc: model configuration dictionary
      training: if the resulting data will be used for training or not
      image_decoder: Image decoder function which returns a tensorflow tensor of the image batch of type tf.float32.     
    Returns:
      A dictionary with all the prerpocessed data for training after anchor ground truth match and data augmentation.
      These data are appropriate for use with a network model.
      All dimensions are normalized with the mc.IMAGE_WIDTH and mc.IMAGE_HEIGHT
  """
  with tf.name_scope("main_input_process"):
    encoded_imgs = record["image/encoded"]
    # all below have shape [BATCH_SIZE, ?]
    xmin = tf.sparse_tensor_to_dense(record["image/object/bbox/xmin"], default_value=0.5)
    xmax = tf.sparse_tensor_to_dense(record["image/object/bbox/xmax"], default_value=0.5)
    ymin = tf.sparse_tensor_to_dense(record["image/object/bbox/ymin"], default_value=0.5)
    ymax = tf.sparse_tensor_to_dense(record["image/object/bbox/ymax"], default_value=0.5)

    standarized_imgs = []
    # NHWC
    new_size = tf.constant([mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH], dtype=tf.int32)
    with tf.name_scope("decode_resize_imgs"):
      for i in range(mc.BATCH_SIZE):
        decoded_img = tf.cast(tf.reverse(image_decoder(encoded_imgs[i], channels=3), axis=[-1]), tf.float32) # RGB -> BGR -> casting
        standarized_imgs.append(
              tf.image.resize_images(decoded_img - tf.constant(mc.BGR_MEANS, tf.float32), size=new_size))
      # resize annotation
      new_size_width = tf.constant(mc.IMAGE_WIDTH, tf.float32, name="image_width")
      new_size_height = tf.constant(mc.IMAGE_HEIGHT, tf.float32, name="image_height")

    if(mc.DATA_AUGMENTATION and training):
      standarized_imgs, xmin, xmax, \
      ymin, ymax = tf_data_augmentation(standarized_imgs, xmin, xmax,
                                        ymin, ymax, mc)
    new_imgs = tf.reshape(
                    tf.stack(standarized_imgs, axis=0),
                    [mc.BATCH_SIZE, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3])
    with tf.name_scope("annotation_resizing"):
      idx = record["image/object/bbox/xmin"].indices

      xmin2 = tf.cast(xmin * new_size_width, tf.float32) # [M]
      new_xmin = _tf_dense_to_sparse_tensor(idx, xmin2)

      xmax2 = tf.cast(xmax * new_size_width, tf.float32)
      new_xmax = _tf_dense_to_sparse_tensor(idx, xmax2)

      ymin2 = tf.cast(ymin * new_size_height, tf.float32)
      new_ymin = _tf_dense_to_sparse_tensor(idx, ymin2)

      ymax2 = tf.cast(ymax * new_size_height, tf.float32)
      new_ymax = _tf_dense_to_sparse_tensor(idx, ymax2)

    with tf.name_scope("anchor_box_search"):
      mc.ANCHOR_BOX = np.array(mc.ANCHOR_BOX)
      s = batch_bbox_transform(mc.ANCHOR_BOX)
      pre_N = np.prod(np.shape(s["xmin"]))
      N = tf.constant(pre_N)
      M = tf.size(new_xmin.values)
      anchor_box_tens = {
        "xmin" : tf.constant(s["xmin"], dtype=tf.float32), # [N]
        "ymin" : tf.constant(s["ymin"], dtype=tf.float32),
        "xmax" : tf.constant(s["xmax"], dtype=tf.float32),
        "ymax" : tf.constant(s["ymax"], dtype=tf.float32)}

      # compute overlaps of bounding boxes and anchor boxes
      dataset_box_tens = {
        "xmin" : tf.cast(new_xmin.values, tf.float32), # [M]
        "ymin" : tf.cast(new_ymin.values, tf.float32),
        "xmax" : tf.cast(new_xmax.values, tf.float32),
        "ymax" : tf.cast(new_ymax.values, tf.float32)}
      
      anchor_box_centers = {"x" : tf.constant(mc.ANCHOR_BOX[:, 0], dtype=tf.float32), 
                            "y" : tf.constant(mc.ANCHOR_BOX[:, 1], dtype=tf.float32),
                            "w" : tf.constant(mc.ANCHOR_BOX[:, 2], dtype=tf.float32),
                            "h" : tf.constant(mc.ANCHOR_BOX[:, 3], dtype=tf.float32)} # 4 tensors of shape [N]
      dataset_box_centers = {"x" : tf.cast(dataset_box_tens["xmax"] + dataset_box_tens["xmin"], tf.float32) * tf.constant(0.5), 
                             "y" : tf.cast(dataset_box_tens["ymax"] + dataset_box_tens["ymin"], tf.float32) * tf.constant(0.5),
                             "w" : tf.cast(dataset_box_tens["xmax"] - dataset_box_tens["xmin"] + tf.constant(1.0), dtype=tf.float32),
                             "h" : tf.cast(dataset_box_tens["ymax"] - dataset_box_tens["ymin"] + tf.constant(1.0), dtype=tf.float32)} # 4 tensors of shape [M]

      aidx = get_matcher_algorithm(mc.MATCHER_ALGORITHM)(record, anchor_box_tens, dataset_box_tens,
                                                         anchor_box_centers, dataset_box_centers, idx,
                                                         N, M, pre_N, mc.BATCH_SIZE)

      with tf.name_scope("deltas"):
        deltas = {
          "dx" : tf.SparseTensor(idx,
                      tf.cast(dataset_box_centers["x"] - tf.gather(anchor_box_centers["x"], indices=aidx.values), dtype=tf.float32) / \
                        tf.gather(anchor_box_centers["w"], indices=aidx.values), dense_shape=new_xmin.dense_shape),
          "dy" : tf.SparseTensor(idx, 
                      tf.cast(dataset_box_centers["y"] - tf.gather(anchor_box_centers["y"], indices=aidx.values), dtype=tf.float32) / \
                        tf.gather(anchor_box_centers["h"], indices=aidx.values), dense_shape=new_xmin.dense_shape),
          "dw" : tf.SparseTensor(idx, tf.log(tf.cast(dataset_box_centers["w"], dtype=tf.float32) / \
                        tf.gather(anchor_box_centers["w"], indices=aidx.values)), dense_shape=new_xmin.dense_shape),
          "dh" : tf.SparseTensor(idx, tf.log(tf.cast(dataset_box_centers["h"], dtype=tf.float32) / \
                        tf.gather(anchor_box_centers["h"], indices=aidx.values)), dense_shape=new_xmin.dense_shape)}

    # return a batch of pre-processed input
    return {
          "image/object/bbox/xmin": new_xmin,
          "image/object/bbox/xmax": new_xmax,
          "image/object/bbox/ymin": new_ymin,
          "image/object/bbox/ymax": new_ymax,
          "image/object/bbox/aidx" : tf.cast(aidx, tf.int64),
          "image/object/bbox/deltas": deltas,
          "image/object/class/label": record["image/object/class/label"],
          "image/decoded": new_imgs,
          "image/height" : record["image/height"],
          "image/width"  : record["image/width"],
          "image/filename": record["image/filename"]}

def tf_data_augmentation(imgs, xmin0, xmax0, ymin0, ymax0, mc):
  """
    This function moves images randomly and in order to keep the same size adds to 0 to the holes.
    To avoid losing a bounding box, there are provided as xmin0, xmax0, ymin0, ymax0.
    Images can also be flipped horizontally.
    With these transformations bounding boxes also change.
    Args:
      imgs: 4D Tensor of images [BATCH_SIZE, W, H, C]
      xmin0: Tensor of xmins of bounding boxes [BATCH_SIZE, ?]
      xmax0: Tensor of xmaxs of bounding boxes [BATCH_SIZE, ?]
      ymin0: Tensor of ymins of bounding boxes [BATCH_SIZE, ?]
      ymax0: Tensor of ymaxs of bounding boxes [BATCH_SIZE, ?]
      mc: model configuration
    Returns:
      unstacked_flipped_imgs: images transformed as a list of length BATCH_SIZE
      new_xmin: new Tensor of xmins of bounding boxes after transormations
      new_xmax: new Tensor of xmaxs of bounding boxes after transormations
      ymin: new Tensor of ymins of bounding boxes after transormations
      ymax: new Tensor of ymaxs of bounding boxes after transormations
  """
  with tf.name_scope("data_augmentation"):
    _zeros = tf.zeros([mc.BATCH_SIZE])
    # distort-move image randomly
    with tf.name_scope("image_distortion"):
      min_drift_x = tf.maximum(-tf.reduce_min(xmin0, axis=1), -float(mc.DRIFT_X)/float(mc.IMAGE_WIDTH)) # for each image
      max_x_per_img = tf.reduce_max(xmax0, axis=1)
      max_drift_x = tf.minimum(1.0 - max_x_per_img, float(mc.DRIFT_X)/float(mc.IMAGE_WIDTH)) # for each image
      min_drift_y = tf.maximum(-tf.reduce_min(ymin0, axis=1), -float(mc.DRIFT_Y)/float(mc.IMAGE_HEIGHT)) # for each image
      max_y_per_img = tf.reduce_max(ymax0, axis=1)
      max_drift_y = tf.minimum(1.0 - max_y_per_img, float(mc.DRIFT_Y)/float(mc.IMAGE_WIDTH)) # for each image

      randoms_dx = tf.random_uniform(shape=[mc.BATCH_SIZE])
      randoms_dy = tf.random_uniform(shape=[mc.BATCH_SIZE])

      unexpanded_dx = randoms_dx * (max_drift_x - min_drift_x) + min_drift_x
      unexpanded_dy = randoms_dy * (max_drift_y - min_drift_y) + min_drift_y
      
      start_x = -tf.minimum(unexpanded_dx, _zeros)
      start_y = -tf.minimum(unexpanded_dy, _zeros)  
      
      begins = tf.stack([tf.cast(start_y * mc.IMAGE_HEIGHT, dtype=tf.int32),
                        tf.cast(start_x * mc.IMAGE_WIDTH, dtype=tf.int32),
                        tf.zeros([mc.BATCH_SIZE], dtype=tf.int32)], axis=-1)
      sizes = tf.stack([tf.cast((1.0 - tf.abs(unexpanded_dy)) * mc.IMAGE_HEIGHT, dtype=tf.int32),
                        tf.cast((1.0 - tf.abs(unexpanded_dx)) * mc.IMAGE_WIDTH, dtype=tf.int32),
                        tf.fill([mc.BATCH_SIZE], -1)], axis=-1)
      
      pad_left = tf.cast(mc.IMAGE_WIDTH * tf.maximum(unexpanded_dx, _zeros), dtype=tf.int32)
      pad_top = tf.cast(mc.IMAGE_HEIGHT * tf.maximum(unexpanded_dy, _zeros), dtype=tf.int32)
      untiled_dx = tf.expand_dims(unexpanded_dx, axis=1)
      untiled_dy = tf.expand_dims(unexpanded_dy, axis=1)

      tile_shape = tf.concat([[1], tf.shape(xmin0)[1:]], axis=0)
      dx = tf.tile(untiled_dx, multiples=tile_shape)
      dy = tf.tile(untiled_dy, multiples=tile_shape)

      xmin = xmin0 + dx
      xmax = xmax0 + dx
      ymin = ymin0 + dy
      ymax = ymax0 + dy

      distorted_imgs = [tf.image.pad_to_bounding_box(
                            tf.slice(imgs[i], begins[i], sizes[i]),
                            pad_top[i], pad_left[i], mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH)
                        for i in range(mc.BATCH_SIZE)]

    # flip image leftright with 50% probability
    with tf.name_scope("flipping"):
      _conds = tf.equal(
                tf.random_uniform(
                    [mc.BATCH_SIZE], minval=0, maxval=2, dtype=tf.int32), 1)
      unstacked_flipped_imgs = []
      unstacked_xmin = tf.unstack(xmin, axis=0, num=mc.BATCH_SIZE)
      unstacked_xmax = tf.unstack(xmax, axis=0, num=mc.BATCH_SIZE)
      for i in range(mc.BATCH_SIZE):
        unstacked_flipped_imgs.append(
          tf.cond(_conds[i], lambda: tf.image.flip_left_right(distorted_imgs[i]), lambda: distorted_imgs[i]))
        u_xmin = unstacked_xmin[i]
        u_xmax = unstacked_xmax[i]
        unstacked_xmin[i] = (tf.cond(_conds[i], lambda: 1.0 - u_xmax, lambda: u_xmin))
        unstacked_xmax[i] = (tf.cond(_conds[i], lambda: 1.0 - u_xmin, lambda: u_xmax))

      # flipped_imgs = tf.stack(unstacked_flipped_imgs, axis=0)
      new_xmin = tf.stack(unstacked_xmin, axis=0)
      new_xmax = tf.stack(unstacked_xmax, axis=0)

  return unstacked_flipped_imgs, new_xmin, new_xmax, ymin, ymax

def get_num_images(mc, keys_to_features, dataset_set="train"):
  """
  This function returns the number of images inside the dataset tfrecord in the path provided my mc
  Args:
    mc: model configuration dictionary
    keys_to_features: keys to features of tfrecord dataset
    dataset_set: The set of the dataset. It can be one of: "train","val","test"
  Returns:
    number of images inside the dataset record
  """
  pr_graph = tf.Graph()
  with pr_graph.as_default():
    pr_sess = tf.Session(graph=pr_graph)
    pr_dataset = tf.contrib.data.make_batched_features_dataset(
                  os.path.join(mc.DATA_PATH, mc.DATASET_NAME.lower() + "_" + dataset_set + ".record"), 
                  1, keys_to_features, num_epochs=1,
                  reader_num_threads=1, parser_num_threads=4, shuffle=False)
    pr_it = pr_dataset.make_one_shot_iterator().get_next()
    num_images = 0
    try:
      while True:
        pr_sess.run(pr_it)
        num_images += 1
    except tf.errors.OutOfRangeError:
      pass
  return num_images

def reduce_dataset_by_class(mc, keys_to_features, dataset_set="train"):
  """
    It creates a reduced dataset of the initial dataset tfrecord which is inside mc["DATA_PATH"].
    The reduced dataset is stored in mc.PREPROCESSED_DATA_DIR if PREPROCESSED_DATA_DIR is defined,
    else it is stored in mc.BASE_DIR. The reduction procedure is done by extracting only the
    classes defined in the mc.CLASSES table. It is important to define the mc.INDICES, so that
    there will be a 1-1 correpsondence between the classes in the reduced dataset and the classes
    in the initial dataset. This is needed because it is easier for the evaluation of the trained
    model.
    Args:
      keys_to_features: keys to features of the initial tfrecord dataset
      dataset_set: The set of the dataset. It can be one of: "train","val","test"
    Returns:
      The number of images inside the reduced dataset
  """
  if("PREPROCESSED_DATA_DIR" in mc):
    _writter_path = os.path.join(mc["PREPROCESSED_DATA_DIR"], "preprocessed_" + mc.DATASET_NAME.lower() + "_" + dataset_set + ".record")
  else:
    _writter_path = os.path.join(mc["BASE_DIR"], "preprocessed_" + mc.DATASET_NAME.lower() + "_" + dataset_set + ".record")
  writer = tf.python_io.TFRecordWriter(_writter_path)
  pr_graph = tf.Graph()
  with pr_graph.as_default():
    pr_sess = tf.Session(graph=pr_graph)
    pr_dataset = tf.contrib.data.make_batched_features_dataset(
                  os.path.join(mc.DATA_PATH, mc.DATASET_NAME.lower() + "_" + dataset_set + ".record"), 
                  1, keys_to_features, num_epochs=1,
                  reader_num_threads=1, parser_num_threads=4, shuffle=False)
    pr_it = pr_dataset.make_one_shot_iterator().get_next()

    # specify tensors to be executed to create the new dataset
    width = tf.reshape(pr_it["image/width"], [])
    height = tf.reshape(pr_it["image/height"], [])
    encoded_image = tf.reshape(pr_it["image/encoded"], [])
    image_filename = tf.reshape(pr_it["image/filename"], [])
    xmin = tf.reshape(pr_it["image/object/bbox/xmin"].values, [-1])
    xmax = tf.reshape(pr_it["image/object/bbox/xmax"].values, [-1])
    ymin = tf.reshape(pr_it["image/object/bbox/ymin"].values, [-1])
    ymax = tf.reshape(pr_it["image/object/bbox/ymax"].values, [-1])
    label = tf.reshape(pr_it["image/object/class/label"].values, [-1])

    # build a reverse mapping for labels 
    reverse_map = {mc.LABEL_INDICES[r_m_idx]: r_m_idx \
                   for r_m_idx in range(len(mc.LABEL_INDICES))}
    # run sessions to fill the data in the dataset with a tf_example per image
    num_images = 0
    try:
      while True:
        temp_height, temp_width,\
        temp_xmin, temp_xmax, temp_ymin, temp_ymax,\
        temp_label,\
        temp_encoded_image,\
        temp_image_filename = pr_sess.run([
                  height, width, xmin, xmax, ymin, ymax,
                  # text,
                  label,
                  encoded_image, image_filename])
        label_indices = [c_label in mc.LABEL_INDICES for c_label in temp_label]
        labels = [reverse_map[c_label] for c_label in temp_label if c_label in mc.LABEL_INDICES]

        tf_example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': dataset_util.int64_feature(temp_height),
          'image/width': dataset_util.int64_feature(temp_width),
          'image/filename': dataset_util.bytes_feature(temp_image_filename.encode('utf8')),
          'image/encoded': dataset_util.bytes_feature(temp_encoded_image),
          'image/object/bbox/xmin': dataset_util.float_list_feature(temp_xmin[label_indices]),
          'image/object/bbox/xmax': dataset_util.float_list_feature(temp_xmax[label_indices]),
          'image/object/bbox/ymin': dataset_util.float_list_feature(temp_ymin[label_indices]),
          'image/object/bbox/ymax': dataset_util.float_list_feature(temp_ymax[label_indices]),
          'image/object/class/label': dataset_util.int64_list_feature(labels),
        }))
        # if corresponding labels found write to dataset file
        if(np.any(label_indices)):
          writer.write(tf_example.SerializeToString())
          num_images += 1
    except tf.errors.OutOfRangeError:
      pass
    writer.close()
    # write num images to a file
    with open(os.path.join(mc.BASE_DIR, "num_images_" + dataset_set + ".txt"), "w") as f:
      f.write(str(num_images))
    # print(RGB_SUM, num_pixels, (RGB_SUM / float(num_pixels)))
  return num_images

def get_initial_anchor_shapes(mc, reduced_dataset_path, keys_to_features):
  """
    Creates the grid of anchors where each anchor contains a default width and height.
    The number of anchors at each grid point are mc.ANCHOR_PER_GRID. The final anchor 
    3D grid is returned as a numpy array.
    Args:
      mc: model configuration
      reduced_dataset_path: The path of the folder where the final dataset resides, after passing all reduction steps (if necessary).
      keys_to_features: keys to features of tfrecord dataset
    Returns:
      The default anchor shapes as numpy array which are going to be used from the final layer of
      the detection algorithm.
  """
  if mc.INIT_ANCHOR_SHAPES["METHOD"] == "KNN":
    k = mc.ANCHOR_PER_GRID
    initial_anchor_shapes = get_k_mean_boxes(k,
                                             reduced_dataset_path, mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT, keys_to_features)
  elif mc.INIT_ANCHOR_SHAPES["METHOD"] == "CONST":
    initial_anchor_shapes = np.array(mc.INIT_ANCHOR_SHAPES["VALUE"])
  return initial_anchor_shapes

def get_k_mean_boxes(k, dataset_path, image_width, image_height, keys_to_features):
  """
  Use all boxes inputs of the dataset as input to k-means
  Args:
    k: stands for k in k-means
    dataset_path: where the dataset is
    mc : model configuration
    keys_to_features: keys to features of tfrecord dataset
  Returns:
    The k 2-D centers
  """
  X = []
  pr_graph = tf.Graph()
  with pr_graph.as_default():
    pr_sess = tf.Session(graph=pr_graph)
    pr_dataset = tf.contrib.data.make_batched_features_dataset(
                  os.path.join(dataset_path), 
                  1, keys_to_features, num_epochs=1,
                  reader_num_threads=1, parser_num_threads=1, shuffle=False)
    pr_it = pr_dataset.make_one_shot_iterator().get_next()

    xmin = tf.reshape(pr_it["image/object/bbox/xmin"].values * tf.cast(pr_it["image/width"], tf.float32), [-1]) 
    xmax = tf.reshape(pr_it["image/object/bbox/xmax"].values * tf.cast(pr_it["image/width"], tf.float32), [-1])
    width = xmax - xmin
    # cx = (xmin + xmax)/2.0
    ymin = tf.reshape(pr_it["image/object/bbox/ymin"].values * tf.cast(pr_it["image/height"], tf.float32), [-1])
    ymax = tf.reshape(pr_it["image/object/bbox/ymax"].values * tf.cast(pr_it["image/height"], tf.float32), [-1])
    # cy = (ymin + ymax)/2.0
    height = ymax - ymin
    try:
      while True:
        temp_width, temp_height = pr_sess.run([
                  width, height])
        X.extend(np.array([temp_width, temp_height]).T)
    except tf.errors.OutOfRangeError:
      pass
  means = np.mean(X, axis=0)
  kmeans = KMeans(n_clusters=k, random_state=0).fit(np.asarray( (X - means)) )

  _centers = kmeans.cluster_centers_
  centers_x = (_centers[:,0] + means[0]) / image_width
  centers_y = (_centers[:,1] + means[1]) / image_height
  centers = np.stack((centers_x, centers_y), axis=1)

  return centers

def get_anchor_box_from_dataset(mc, reduced_dataset_path, keys_to_features):
  """
    The anchors are defined by the dataset
  """
  # get initial anchor shapes based on config
  mc.INITIAL_ANCHOR_SHAPES = get_initial_anchor_shapes(mc, reduced_dataset_path, keys_to_features)
  
  anchor_box = np.array(config_cooker.set_anchors(mc))
  anchors = len(anchor_box)

  return anchor_box, anchors