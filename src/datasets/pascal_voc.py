import numpy as np
from six.moves import xrange
import os
import sys
from shutil import rmtree
from matplotlib import pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import tensorflow as tf
import json
from easydict import EasyDict as edict
from config import config_cooker
from utils.tf_util import batch_iou, batch_bbox_transform, compute_distances, _tf_dense_to_sparse_tensor
import utils.util as util
import creation.dataset_util as dataset_util
from voc_eval import voc_eval
EPSILON = 0
import imdb

# def _parser(batch_records):
def pascal_voc(mc, train_graph, eval_graph):
  with tf.name_scope("PASCAL_VOC_input") as scope:
    keys_to_features = imdb.get_keys_to_features()
    # set initial record paths to read data from
    dataset_train_path = os.path.join(mc.DATA_PATH, "pascal_voc_train.record")
    dataset_eval_path = os.path.join(mc.DATA_PATH, "pascal_voc_val.record")

    mc.ANCHOR_BOX, mc.ANCHORS = imdb.get_anchor_box_from_dataset(mc, dataset_train_path, keys_to_features)

    # create a new dataset with preprocessed/filtered records
    if(mc.REDUCE_DATASET):
      if(not mc.ALREADY_PREPROCESSED):
        imdb.reduce_dataset_by_class(mc, keys_to_features, dataset_set="train")
        if(eval_graph):
          eval_mc = edict(mc.copy())
          # eval_mc.BATCH_SIZE = 1
          eval_mc.IS_TRAINING = False
          eval_mc.DATA_AUGMENTATION = False
          mc.EVAL_ITERS = imdb.reduce_dataset_by_class(eval_mc, keys_to_features, dataset_set="val")
          eval_mc.EVAL_ITERS = mc.EVAL_ITERS
          print("EVAL ITERS :%d"%(mc.EVAL_ITERS))
      else:
        pass
        # with open(os.path.join(mc.TRAIN_DIR, "BGR_MEANS.txt"), "r") as f:
        #   mc.BGR_MEANS = np.fromstring(f.readline().split("[[[")[1].split("]]]")[0], sep =" ")
      if(mc.PREPROCESSED_DATA_DIR):
        dataset_train_path = os.path.join(mc.PREPROCESSED_DATA_DIR, "preprocessed_pascal_voc_train.record")
        dataset_eval_path = os.path.join(mc.PREPROCESSED_DATA_DIR, "preprocessed_pascal_voc_val.record")
      else:
        dataset_train_path = os.path.join(mc.TRAIN_DIR, "preprocessed_pascal_voc_train.record")
        dataset_eval_path = os.path.join(mc.TRAIN_DIR, "preprocessed_pascal_voc_val.record")
    elif eval_graph:
        mc.EVAL_ITERS = imdb.get_num_images(mc, keys_to_features, dataset_set="val")
    # prepare training dataset
    if train_graph:
      with train_graph.as_default():
        dataset_train = tf.contrib.data.make_batched_features_dataset(dataset_train_path, 
              mc.BATCH_SIZE, keys_to_features, num_epochs=None,
              reader_num_threads=mc.NUM_THREADS/2, parser_num_threads=mc.NUM_THREADS/2, shuffle_buffer_size=12000, shuffle=True,
              sloppy_ordering=True)
        it_train = dataset_train.make_one_shot_iterator()
        train_list = imdb.load_data(it_train.get_next(), mc, training=True, image_decoder=tf.image.decode_jpeg)
    else:
      train_list = None
    # prepare evaluation dataset
    if(eval_graph):
      with eval_graph.as_default():
        dataset_eval = tf.contrib.data.make_batched_features_dataset(dataset_eval_path,
              mc.BATCH_SIZE, keys_to_features, num_epochs=None,
              reader_num_threads=mc.NUM_THREADS/2, parser_num_threads=mc.NUM_THREADS/2, shuffle=False)
        it_eval = dataset_eval.make_one_shot_iterator()
        eval_mc = edict(mc.copy())
        eval_mc.IS_TRAINING = False
        eval_mc.DATA_AUGMENTATION = False
        eval_list = imdb.load_data(it_eval.get_next(), eval_mc, training=False, image_decoder=tf.image.decode_jpeg)
    else:
      eval_list = None
    
    return train_list, eval_list, mc


def evaluate_detections(mc, eval_dir, global_step, all_boxes, img_names_raw):
  """Evaluate detection results.
  Args:
    eval_dir: directory to write evaluation logs
    global_step: step of the checkpoint
    all_boxes: all_boxes[cls][image_idx] = N x 5 arrays of 
      [xmin, ymin, xmax, ymax, score]
    img_names_raw: raw name of all images
  Returns:
    aps: array of average precisions.
    names: class names corresponding to each ap
  """
  img_names = [name.split(".")[0] for name in img_names_raw]

  # det_file_dir = os.path.join(
  #     eval_dir, 'detection_files_{:s}'.format(global_step))
  if not os.path.isdir(eval_dir):
    os.mkdir(eval_dir)
  det_file_path_template = os.path.join(eval_dir, '{:s}.txt')

  for cls_idx, cls in enumerate(mc.CLASS_NAMES):
    det_file_name = det_file_path_template.format(cls)
    with open(det_file_name, 'wt') as f:
      for im_idx in xrange(len(img_names)):
        dets = all_boxes[cls_idx][im_idx]
        # VOC expects 1-based indices
        for k in xrange(len(dets)):
          f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
              format(img_names[im_idx], dets[k][-1], 
                      dets[k][0]+1, dets[k][1]+1,
                      dets[k][2]+1, dets[k][3]+1)
          )

  # Evaluate detection results
  annopath = os.path.join(
      mc.DATA_PATH,
      'VOC2012',
      'Annotations',
      '{:s}.xml'
  )
  # imagesetfile = os.path.join(
  #     mc.DATA_PATH,
  #     'VOC2012',
  #     'ImageSets',
  #     'Main',
  #     'val' + '.txt'
  # )
  cachedir = os.path.join(mc["EVAL_DIR"], 'annotations_cache')
  # remove cache folder before calculating all the aps
  if(os.path.exists(cachedir)):
    rmtree(cachedir)
  aps = []
  for i, cls in enumerate(mc.CLASS_NAMES):
    filename = det_file_path_template.format(cls)
    _,  _, ap = voc_eval(
        filename, annopath, img_names, cls, cachedir, ovthresh=0.5,
        use_07_metric=False)
    aps += [ap]
    # print ('{:s}: AP = {:.4f}'.format(cls, ap))

  # print ('Mean AP = {:.4f}'.format(np.mean(aps)))
  return aps, None