import os
import sys
import json
import imdb
import tensorflow as tf
import numpy as np
from easydict import EasyDict as edict
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from config import config_cooker
from utils.tf_util import batch_iou
import utils.util as util

def coco(mc, train_graph, eval_graph):
  with tf.name_scope("COCO_input"):
    keys_to_features = imdb.get_keys_to_features()
    
    # return tf.parse_example(batch_records, keys_to_features)
    dataset_train_path = os.path.join(mc.DATA_PATH, "coco_train.record")
    dataset_eval_path = os.path.join(mc.DATA_PATH, "coco_val.record")
    
    # create a new dataset with preprocessed/filtered records
    if(mc.REDUCE_DATASET and not mc.ALREADY_PREPROCESSED):
      imdb.reduce_dataset_by_class(mc, keys_to_features, dataset_set="train")
      if(eval_graph):
        eval_mc = edict(mc.copy())
        # eval_mc.BATCH_SIZE = 1
        eval_mc.IS_TRAINING = False
        eval_mc.DATA_AUGMENTATION = False
        mc.EVAL_ITERS = imdb.reduce_dataset_by_class(eval_mc, keys_to_features, dataset_set="val")
        eval_mc.EVAL_ITERS = mc.EVAL_ITERS
        dataset_train_path = os.path.join(mc["BASE_DIR"], "preprocessed_" + mc.DATASET_NAME.lower() + "_train.record")
        dataset_eval_path = os.path.join(mc["BASE_DIR"], "preprocessed_" + mc.DATASET_NAME.lower() + "_val.record")
        print("EVAL ITERS :%d"%(mc.EVAL_ITERS))
    
    if(mc.REDUCE_DATASET and mc.PREPROCESSED_DATA_DIR):
      dataset_train_path = os.path.join(mc["PREPROCESSED_DATA_DIR"], "preprocessed_" + mc.DATASET_NAME.lower() + "_train.record")
      dataset_eval_path = os.path.join(mc["PREPROCESSED_DATA_DIR"], "preprocessed_" + mc.DATASET_NAME.lower() + "_val.record")
    
    # get anchor boxes before creating the input graph
    mc.ANCHOR_BOX, mc.ANCHORS = imdb.get_anchor_box_from_dataset(mc, dataset_train_path, keys_to_features)

    # prepare training dataset
    if train_graph:
      with train_graph.as_default():
        dataset_train = tf.contrib.data.make_batched_features_dataset(dataset_train_path, 
              mc.BATCH_SIZE, keys_to_features, num_epochs=mc.TRAIN_EPOCHS,
              reader_num_threads=8, parser_num_threads=8, shuffle_buffer_size=13000 if mc.IS_TRAINING else 512, sloppy_ordering=True)
        it_train = dataset_train.make_one_shot_iterator()
        train_list = imdb.load_data(it_train.get_next(), mc, training=True, image_decoder=tf.image.decode_jpeg)
    else:
      train_list = None
    
    # prepare evaluation dataset
    if eval_graph:
      with eval_graph.as_default():
        eval_mc = edict(mc.copy())
        # eval_mc.BATCH_SIZE = 1
        eval_mc.IS_TRAINING = False
        eval_mc.DATA_AUGMENTATION = False
        dataset_eval = tf.contrib.data.make_batched_features_dataset(dataset_eval_path, 
               eval_mc.BATCH_SIZE, keys_to_features, num_epochs=None,
              reader_num_threads=8, parser_num_threads=8, shuffle=False, drop_final_batch=True)
        it_eval = dataset_eval.make_one_shot_iterator()
        eval_list = imdb.load_data(it_eval.get_next(), eval_mc, training=False, image_decoder=tf.image.decode_png)
    else:
      eval_list = None

  return train_list, eval_list, mc

def evaluate_detections(mc, eval_dir, global_step, all_boxes, filenames):
  """Evaluate detection results.
  Args:
    eval_dir: directory to write evaluation logs
    global_step: step of the checkpoint
    all_boxes: all_boxes[cls][image_idx] = N x 5 arrays of 
      [xmin, ymin, xmax, ymax, score]
    img_names_raw: raw name of all images
  Returns:
    mAP: medium average precision.
    names: None, no names returned
  """
  annType = "bbox"
  cocoGt = COCO(os.path.join(mc.DATA_PATH, "annotations", "%s_%s.json" %("instances", "val2017")))
  result = []
  for cls in range(mc.CLASSES):
    for image_idx, bbox in enumerate(all_boxes[cls]):
      result.append({"image_id": image_idx, "category_id" : (cls + 1), "bbox": bbox[:4], "score": bbox[4]})
  temporaryResultsFile = os.path.join(eval_dir, "result_annotations.json")
  with open(temporaryResultsFile, "w") as f:
    json.dump(result, f)
  cocoDt = cocoGt.loadRes(temporaryResultsFile)
  cocoEval = COCOeval(cocoGt, cocoDt, annType)
  imgIds = sorted(cocoGt.getImgIds())
  cocoEval.params.imgIds  = imgIds
  cocoEval.evaluate()
  cocoEval.accumulate()
  cocoEval.summarize()
  # get mean precision
  mAP = [cocoEval.stats[2]]
  return mAP, None

def main(_):
  load_precomputed = False
  if load_precomputed:
    mc = edict(json.load(open("precomputed_mc.json", "r")))
    mc.ANCHOR_BOX = np.array(mc.ANCHOR_BOX)
    mc.NUM_EPOCHS = 100
    mc.SUMMARY_STEP = 1000
    mc.DATA_AUGMENTATION = False
  else:
    mc = config_cooker.cook_config("/media/terabyte/projects/Thesis/transfer_learning/auto_ITL/scripts/coco_tests/coco_squeezeDet_config.json")
    mc.NUM_EPOCHS = 100
    mc.SUMMARY_STEP = 1000
    mc.DATA_AUGMENTATION = False
    mc.ANCHOR_BOX = np.array(mc.ANCHOR_BOX)
    mc_copy = mc
    mc_copy.ANCHOR_BOX = mc_copy.ANCHOR_BOX.tolist()
    json.dump(mc_copy, open("precomputed_mc.json", "w"))
  t_g, e_g = tf.Graph(), tf.Graph()
  mc["TRAIN_DIR"] = "/media/terabyte/projects/Thesis/trainings/coco_TRAIN_DIR1"
  train_input, eval_input, mc = coco(mc, t_g, e_g)
  with tf.Session(graph=t_g) as sess:
    X = sess.run(train_input)
    print(X)

if __name__ == "__main__":
  tf.app.run()