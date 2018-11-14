import os
import sys
import subprocess
import json
import imdb
import tensorflow as tf
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from easydict import EasyDict as edict
from config import config_cooker
from utils.tf_util import batch_iou
import utils.util as util


def kitti(mc, train_graph, eval_graph):
  with tf.name_scope("KITTI_input"):
    keys_to_features = imdb.get_keys_to_features()

    dataset_train_path = os.path.join(mc.DATA_PATH, "kitti_train.record")
    dataset_eval_path = os.path.join(mc.DATA_PATH, "kitti_val.record")
    
    # get anchor boxes before creating the input graph
    mc.ANCHOR_BOX, mc.ANCHORS = imdb.get_anchor_box_from_dataset(mc, dataset_train_path, keys_to_features)
    
    # prepare training dataset
    if train_graph:
      with train_graph.as_default():
        dataset_train = tf.contrib.data.make_batched_features_dataset(dataset_train_path,
              mc.BATCH_SIZE, keys_to_features, num_epochs=None,
              reader_num_threads=mc.NUM_THREADS/2, parser_num_threads=mc.NUM_THREADS/2, shuffle_buffer_size=1200 if mc.IS_TRAINING else 512, sloppy_ordering=True)
        it_train = dataset_train.make_one_shot_iterator()
        train_list = imdb.load_data(it_train.get_next(), mc, training=True, image_decoder=tf.image.decode_png)
    else:
      train_list = None
    
    # prepare evaluation dataset
    if eval_graph:
      with eval_graph.as_default():
        eval_mc = edict(mc.copy())
        eval_mc.IS_TRAINING = False
        eval_mc.DATA_AUGMENTATION = False
        dataset_eval = tf.contrib.data.make_batched_features_dataset(dataset_eval_path,
               eval_mc.BATCH_SIZE, keys_to_features, num_epochs=None,
              reader_num_threads=mc.NUM_THREADS/2, parser_num_threads=mc.NUM_THREADS/2, shuffle=False, drop_final_batch=True)
        it_eval = dataset_eval.make_one_shot_iterator()
        eval_list = imdb.load_data(it_eval.get_next(), eval_mc, training=False, image_decoder=tf.image.decode_png)
    else:
      eval_list = None
  
  mc.EVAL_TOOL_PATH = os.path.join(os.path.dirname(__file__), "kitti-eval/cpp/evaluate_object")
  
  return train_list, eval_list, mc

def evaluate_detections(mc, eval_dir, global_step, all_boxes, filenames):
  """Evaluate detection results.
  Args:
    eval_dir: directory to write evaluation logs
    global_step: step of the checkpoint
    all_boxes: all_boxes[cls][image] = N x 5 arrays of 
      [xmin, ymin, xmax, ymax, score]
  Returns:
    aps: array of average precisions.
    analysis: Detection Analysis dir
  """
  det_file_dir = os.path.join(
      eval_dir, 'detection_files_{:s}'.format(global_step), 'data')
  if not os.path.isdir(det_file_dir):
    os.makedirs(det_file_dir)
  
  unextended_filenames = [os.path.basename(fname).split(".")[0] for fname in filenames]

  for im_idx, index in enumerate(unextended_filenames):
    filename = os.path.join(det_file_dir, index + '.txt')
    with open(filename, 'wt') as f:
      for cls_idx, cls in enumerate(mc.CLASS_NAMES):
        dets = all_boxes[cls_idx][im_idx]
        for k in xrange(len(dets)):
          f.write(
              '{:s} -1 -1 0.0 {:.2f} {:.2f} {:.2f} {:.2f} 0.0 0.0 0.0 0.0 0.0 '
              '0.0 0.0 {:.3f}\n'.format(
                  cls.lower(), dets[k][0], dets[k][1], dets[k][2], dets[k][3],
                  dets[k][4])
          )

  eval_list_file_path = os.path.join(mc.DATA_PATH, 'eval.txt')
  with open(eval_list_file_path, "w") as f:
    for fname in unextended_filenames:
      f.write(fname + "\n")

  cmd = mc.EVAL_TOOL_PATH + ' ' \
        + os.path.join(mc.DATA_PATH, 'training') + ' ' \
        + eval_list_file_path + ' ' \
        + os.path.dirname(det_file_dir) + ' ' + str(len(unextended_filenames))

  print('Running: {}'.format(cmd))
  status = subprocess.call(cmd, shell=True)

  aps = []
  names = []
  for cls in mc.CLASS_NAMES:
    det_file_name = os.path.join(
        os.path.dirname(det_file_dir), 'stats_{:s}_ap.txt'.format(cls))
    if os.path.exists(det_file_name):
      with open(det_file_name, 'r') as f:
        lines = f.readlines()
      assert len(lines) == 3, \
          'Line number of {} should be 3'.format(det_file_name)

      aps.append(float(lines[0].split('=')[1].strip()))
      aps.append(float(lines[1].split('=')[1].strip()))
      aps.append(float(lines[2].split('=')[1].strip()))
    else:
      aps.extend([0.0, 0.0, 0.0])

    names.append(cls+'_easy')
    names.append(cls+'_medium')
    names.append(cls+'_hard')
  det_error_file = os.path.join(det_file_dir, "det_error_file.txt")
  analysis  = analyze_detections(mc, det_file_dir, det_error_file, unextended_filenames)
  return aps, analysis

def analyze_detections(mc, detection_file_dir, det_error_file, filenames):
  def _save_detection(f, idx, error_type, det, score):
    f.write(
        '{:s} {:s} {:.1f} {:.1f} {:.1f} {:.1f} {:s} {:.3f}\n'.format(
            idx, error_type,
            det[0]-det[2]/2., det[1]-det[3]/2.,
            det[0]+det[2]/2., det[1]+det[3]/2.,
            mc.CLASS_NAMES[int(det[4])], 
            score
        )
    )
  class_to_idx = dict(zip(mc.CLASS_NAMES, xrange(mc.CLASSES)))
  # load detections
  _det_rois = {}
  for idx in filenames:
    det_file_name = os.path.join(detection_file_dir, idx+'.txt')
    with open(det_file_name) as f:
      lines = f.readlines()
    f.close()
    bboxes = []
    for line in lines:
      obj = line.strip().split(' ')
      cls = class_to_idx[obj[0].lower().strip()]
      xmin = float(obj[4])
      ymin = float(obj[5])
      xmax = float(obj[6])
      ymax = float(obj[7])
      score = float(obj[-1])

      x, y, w, h = util.bbox_transform_inv([xmin, ymin, xmax, ymax])
      bboxes.append([x, y, w, h, cls, score])
    bboxes.sort(key=lambda x: x[-1], reverse=True)
    _det_rois[idx] = bboxes

  # do error analysis
  num_objs = 0.
  num_dets = 0.
  num_correct = 0.
  num_loc_error = 0.
  num_cls_error = 0.
  num_bg_error = 0.
  num_repeated_error = 0.
  num_detected_obj = 0.
  _rois = _load_kitti_annotation(mc, filenames, class_to_idx)
  with open(det_error_file, 'w') as f:
    for idx in filenames:
      gt_bboxes = np.array(_rois[idx])
      num_objs += len(gt_bboxes)
      detected = [False]*len(gt_bboxes)

      det_bboxes = _det_rois[idx]
      if len(gt_bboxes) < 1:
        continue

      for i, det in enumerate(det_bboxes):
        if i < len(gt_bboxes):
          num_dets += 1
        ious = util.batch_iou(gt_bboxes[:, :4], det[:4])
        max_iou = np.max(ious)
        gt_idx = np.argmax(ious)
        if max_iou > 0.1:
          if gt_bboxes[gt_idx, 4] == det[4]:
            if max_iou >= 0.5:
              if i < len(gt_bboxes):
                if not detected[gt_idx]:
                  num_correct += 1
                  detected[gt_idx] = True
                else:
                  num_repeated_error += 1
            else:
              if i < len(gt_bboxes):
                num_loc_error += 1
                _save_detection(f, idx, 'loc', det, det[5])
          else:
            if i < len(gt_bboxes):
              print(gt_bboxes[gt_idx, 4], det[4])
              num_cls_error += 1
              _save_detection(f, idx, 'cls', det, det[5])
        else:
          if i < len(gt_bboxes):
            num_bg_error += 1
            _save_detection(f, idx, 'bg', det, det[5])

      for i, gt in enumerate(gt_bboxes):
        if not detected[i]:
          _save_detection(f, idx, 'missed', gt, -1.0)
      num_detected_obj += sum(detected)
  f.close()

  print ('Detection Analysis:')
  print ('    Number of detections: {}'.format(num_dets))
  print ('    Number of objects: {}'.format(num_objs))
  print ('    Percentage of correct detections: {}'.format(
    num_correct/num_dets))
  print ('    Percentage of localization error: {}'.format(
    num_loc_error/num_dets))
  print ('    Percentage of classification error: {}'.format(
    num_cls_error/num_dets))
  print ('    Percentage of background error: {}'.format(
    num_bg_error/num_dets))
  print ('    Percentage of repeated detections: {}'.format(
    num_repeated_error/num_dets))
  print ('    Recall: {}'.format(
    num_detected_obj/num_objs))

  out = {}
  out['num of detections'] = num_dets
  out['num of objects'] = num_objs
  out['% correct detections'] = num_correct/num_dets
  out['% localization error'] = num_loc_error/num_dets
  out['% classification error'] = num_cls_error/num_dets
  out['% background error'] = num_bg_error/num_dets
  out['% repeated error'] = num_repeated_error/num_dets
  out['% recall'] = num_detected_obj/num_objs

  return out

def _load_kitti_annotation(mc, filenames, class_to_idx):
  def _get_obj_level(obj):
    height = float(obj[7]) - float(obj[5]) + 1
    truncation = float(obj[1])
    occlusion = float(obj[2])
    if height >= 40 and truncation <= 0.15 and occlusion <= 0:
        return 1
    elif height >= 25 and truncation <= 0.3 and occlusion <= 1:
        return 2
    elif height >= 25 and truncation <= 0.5 and occlusion <= 2:
        return 3
    else:
        return 4
  label_path = os.path.join(mc.DATA_PATH, 'training', 'label_2')
  idx2annotation = {}
  for index in filenames:
    filename = os.path.join(label_path, index+'.txt')
    with open(filename, 'r') as f:
      lines = f.readlines()
    f.close()
    bboxes = []
    for line in lines:
      obj = line.strip().split(' ')
      try:
        cls = class_to_idx[obj[0].lower().strip()]
      except:
        continue

      if mc.EXCLUDE_HARD_EXAMPLES and _get_obj_level(obj) > 3:
        continue
      xmin = float(obj[4])
      ymin = float(obj[5])
      xmax = float(obj[6])
      ymax = float(obj[7])
      assert xmin >= 0.0 and xmin <= xmax, \
          'Invalid bounding box x-coord xmin {} or xmax {} at {}.txt' \
              .format(xmin, xmax, index)
      assert ymin >= 0.0 and ymin <= ymax, \
          'Invalid bounding box y-coord ymin {} or ymax {} at {}.txt' \
              .format(ymin, ymax, index)
      x, y, w, h = util.bbox_transform_inv([xmin, ymin, xmax, ymax])
      bboxes.append([x, y, w, h, cls])

    idx2annotation[index] = bboxes

  return idx2annotation



def main(_):
  load_precomputed = False
  if load_precomputed:
    mc = edict(json.load(open("precomputed_mc.json", "r")))
    mc.ANCHOR_BOX = np.array(mc.ANCHOR_BOX)
    mc.NUM_EPOCHS = 1
    mc.SUMMARY_STEP = 1000
    mc.DATA_AUGMENTATION = False
  else:
    mc = config_cooker.cook_config("/media/terabyte/projects/Thesis/transfer_learning/auto_ITL/scripts/kitti_tests/kitti_squeezeDet_config.json")
    mc.NUM_EPOCHS = 1
    mc.SUMMARY_STEP = 1000
    mc.DATA_AUGMENTATION = False
    mc.ANCHOR_BOX = np.array(mc.ANCHOR_BOX)
    mc_copy = mc
    mc_copy.ANCHOR_BOX = mc_copy.ANCHOR_BOX.tolist()
    json.dump(mc_copy, open("precomputed_mc.json", "w"))
  t_g, e_g = tf.Graph(), None
  mc["TRAIN_DIR"] = "/media/terabyte/projects/Thesis/trainings/kitti_TRAIN_DIR1"
  train_input, eval_input, mc = kitti(mc, t_g, e_g)

  with tf.Session(graph=t_g) as sess:
    X = sess.run(train_input['image/object/class/label'])
    print(X)

if __name__ == "__main__":
  tf.app.run()