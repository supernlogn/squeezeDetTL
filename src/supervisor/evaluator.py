import os
import sys
import json
import numpy as np
from six.moves import xrange

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import datasets
import nets
import tensorflow as tf


def create_evaluation_model(mc, eval_input, eval_graph):
  """
    Create an evaluation Session and operations
    to be computed for the evaluation of the model.
  """
  eval_mc = mc
  if not "EVAL_DIR" in mc:
    mc.EVAL_DIR = os.path.join(mc["BASE_DIR"], "evals")
    mc["EVAL_DIR"] = os.path.join(mc["BASE_DIR"], "evals")

  if not os.path.exists(eval_mc["EVAL_DIR"]):
    os.mkdir(eval_mc["EVAL_DIR"])
  
  eval_mc.IS_TRAINING = False
  eval_mc.DATA_AUGMENTATION = False
  
  eval_sess = tf.Session(graph=eval_graph, config=tf.ConfigProto(allow_soft_placement=True))
  
  with eval_graph.as_default():
    eval_model = nets.get_net(eval_mc.NET)(eval_mc, eval_input, global_step=True)
    eval_sess.run(tf.global_variables_initializer())
    eval_ops = eval_model.geteval_op_list()
    if eval_model.viz_op != None:
      summary_op = tf.summary.merge([tf.summary.merge_all(), eval_model.viz_op])
    else:
      summary_op = tf.summary.merge([tf.summary.merge_all()])
    eval_writer = tf.summary.FileWriter(eval_mc["EVAL_DIR"], graph=eval_graph, flush_secs=100000000)
    eval_ops.append(summary_op)
    eval_saver = tf.train.Saver()
  
  return eval_sess, eval_ops, eval_saver, eval_writer

def evaluate(mc, eval_graph, eval_sess, eval_ops, eval_saver, eval_writer):
  """
    Evaluate the model obtained from the last checkpoint
    and create the tensorboard summaries.
  """
  ckpt = tf.train.get_checkpoint_state(mc["TRAIN_DIR"])
  # global_vars = tf.global_variables()
  # is_not_initialized = eval_sess.run([tf.is_variable_initialized(var) for var in global_vars if(not "iou" in var)])
  eval_saver.restore(eval_sess, ckpt.model_checkpoint_path)
  evaluation_func = datasets.get_evaluation_func(mc.DATASET_NAME)
  # get current training global step from last checkpoint
  with open(os.path.join(mc["TRAIN_DIR"], "checkpoint")) as f:
    latest_checkpoint = f.readline().split("model_checkpoint_path: \"")[-1][:-2]
  global_step = latest_checkpoint.split("/")[-1].split(".data")[0].split("-")[-1]
  g_s = int(global_step)
  # fill data from iterations to feed them to the evaluation tool
  prediction_boxes, score, cls_idx_per_img, img_names, widths, heights = [], [], [], [], [], []
 
  max_iters = int(mc.EVAL_ITERS / mc.BATCH_SIZE)
  all_boxes = [[[] for _ in xrange(max_iters * mc.BATCH_SIZE)]
                for _ in xrange(mc.CLASSES)]
  for i in xrange(max_iters):
    res = eval_sess.run(eval_ops)
    prediction_boxes.append(res[0])
    score.append(res[1])
    cls_idx_per_img.append(res[2])
    img_names.extend(res[3])
    widths.extend(res[4])
    heights.extend(res[5])
    eval_writer.add_summary(res[-1], g_s)
    print("%d/%d"%(i, max_iters))

  # loop per step/batch
  local_iter = 0
  for eval_iter in xrange(max_iters):
    # loop per image inside the batch
    for c, b, s in zip(cls_idx_per_img[eval_iter],\
                      prediction_boxes[eval_iter],\
                      score[eval_iter]):
      x_scale = widths[local_iter] / float(mc.IMAGE_WIDTH)
      y_scale = heights[local_iter] / float(mc.IMAGE_HEIGHT)
      for i, cls_idx in enumerate(c):
        c_i = cls_idx
        all_boxes[c_i][local_iter].append([b["xmins"][i] * x_scale,
                                               b["ymins"][i] * y_scale,
                                               b["xmaxs"][i] * x_scale,
                                               b["ymaxs"][i] * y_scale, s[i]])
      local_iter += 1
  # analyse
  aps, analysis = evaluation_func(mc, mc["EVAL_DIR"], global_step, all_boxes, img_names)
  # write analysis
  if(analysis):
    with open(os.path.join(mc["EVAL_DIR"], "analysis_raw.txt"), "a") as f:
      f.write(str(global_step) + "\n{\n")
      for el in analysis.keys():
        f.write(str(el) + ": " + str(analysis[el]) + "\n")
      f.write("}\n")
  # write mAPs
  if os.path.exists(os.path.join(mc["EVAL_DIR"], "mAP_history.json")):
    history = json.load(open(os.path.join(mc["EVAL_DIR"], "mAP_history.json")))["history"]
  else:
    history = []
  with open(os.path.join(mc["EVAL_DIR"], "mAP_history.json"), "w") as f:
    history.append({"step": global_step, "mAP": np.mean(aps)})
    json.dump({"history": history}, f)
    # f.write("{\n\"step\": "+global_step+",\n\"mAP\": {:.4f}\n".format(np.mean(aps)))
    # for cls_idx, cls in enumerate(mc.CLASS_NAMES):
    #   f.write("\"{:s}_AP\" : \"e\": {:.4f}, \"m\": {:.4f}, \"h\": {:.4f}\n".format(cls, aps[cls_idx*3], aps[cls_idx*3+1], aps[cls_idx*3+2]))
    # f.write("}\n")
  print("evaluation done for" + global_step)
  # flush eval_writer
  eval_writer.flush()

def evaluate_once(mc):
  """ Create an evalution graph and evaluate it once.
      A previous training is required to create a valid checkpoint from which
      to load the trained weights.
      Args:
        mc: model configuration 
  """
  # allocate evaluation graph and initialize as empty
  eval_graph = tf.Graph()
  # create dataset input handlers for the evaluation
  _, eval_list, mc = datasets.get_dataset(mc.DATASET_NAME)(mc,
                                                           None,
                                                           eval_graph)
  # add model to evaluation graph
  eval_sess, eval_ops, eval_saver, eval_writer =\
        create_evaluation_model(mc, eval_list, eval_graph)
  
  # evaluate
  evaluate(mc, eval_graph, eval_sess, eval_ops, eval_saver, eval_writer)
  
  return True