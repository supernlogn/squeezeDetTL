from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import os.path
from six.moves import xrange
from easydict import EasyDict as edict

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform
from supervisor.evaluator import create_evaluation_model, evaluate
import datasets
import nets

def train(mc):
  """
    It trains the SqueezeDet model.
    Args:
      mc: The model configuration as an EasyDict object.
    Returns:
      Always True
  """

  if not "TRAIN_DIR" in mc:
    mc.TRAIN_DIR = os.path.join(mc["BASE_DIR"], "train")
    mc["TRAIN_DIR"] = os.path.join(mc["BASE_DIR"], "train")

  # if it does not exist in the filesystem, create it. 
  if(not os.path.exists(mc["TRAIN_DIR"])):
    os.mkdir(mc["TRAIN_DIR"])

  # create a training and evaluation graphs
  train_graph,\
  eval_graph = tf.Graph(),\
               tf.Graph() if (mc.EVAL_WITH_TRAIN) else None

  # get training and evaluation inputs
  train_list, eval_list, mc = datasets.get_dataset(mc.DATASET_NAME)(mc,
                                                                train_graph,
                                                                eval_graph)
  # get checkpoint
  if("ckpt_path" in mc):
    ckpt = tf.train.get_checkpoint_state(mc["ckpt_path"])
  else:
    ckpt = tf.train.get_checkpoint_state(mc["TRAIN_DIR"])

  
  with train_graph.as_default():
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    jit_level = tf.OptimizerOptions.ON_2
    sess_config.graph_options.optimizer_options.global_jit_level = jit_level
    if(mc["SAVE_XLA_TIMELINE"]):
      run_metadata = tf.RunMetadata()
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    else:
      run_metadata = None
      run_options = None
    # get/set global step
    global_step = tf.train.get_or_create_global_step()
    model = nets.get_net(mc.NET)(mc, train_list, global_step=global_step)
 
    if mc.VISUALIZE_ON:
      summary_op = tf.summary.merge([tf.summary.merge_all(), model.viz_op])
    else:
      summary_op = tf.summary.merge([tf.summary.merge_all()])

    if(not mc.LOAD_PRETRAINED_MODEL):
      var_list = tf.all_variables()
      if(mc.LAYERS_TO_LOAD):
        new_var_list = [var_name for var_name in var_list if np.any([layer_name+"/" in str(var_name) for layer_name in mc.LAYERS_TO_LOAD])]
      else:
        new_var_list = var_list
      train_saver = tf.train.Saver(var_list=new_var_list)
    train_sess = tf.train.MonitoredTrainingSession(
      master='',
      is_chief=True,
      checkpoint_dir=mc["TRAIN_DIR"],
      scaffold=tf.train.Scaffold(init_op=tf.global_variables_initializer() 
                                        if mc.LOAD_PRETRAINED_MODEL else None, 
                                summary_op=summary_op),
      hooks=None,
      chief_only_hooks=None,
      save_summaries_steps=mc["SUMMARY_STEP"],
      config=sess_config,
      stop_grace_period_secs=120,
      log_step_count_steps=100,
      max_wait_secs=7200,
      save_checkpoint_steps=mc["checkpoint_step"])
    train_writer = tf.summary.FileWriter(mc["TRAIN_DIR"], train_sess.graph)
  if(mc.EVAL_WITH_TRAIN):
    eval_sess, eval_ops, eval_saver, eval_writer =\
            create_evaluation_model(mc, eval_list, eval_graph)
  
  # load variables from checkpoint
  if(not mc.LOAD_PRETRAINED_MODEL and ckpt and ckpt.model_checkpoint_path):
    train_saver.restore(train_sess, ckpt.model_checkpoint_path)
  
  # get current step
  g_s = train_sess.run(global_step, options=run_options)
  # run the whole thing
  for i in xrange(mc["MAX_STEPS"]):
    train_sess.run(model.train_op,
                   run_metadata=run_metadata, options=run_options)
    if i % mc["EVAL_PER_STEPS"] == 0 and i >= 5000 and mc.EVAL_WITH_TRAIN:
      # run evaluation
      evaluate(mc, eval_graph, eval_sess, eval_ops, eval_saver, eval_writer)
    elif i % mc["EVAL_PER_STEPS"] == 5 and (g_s + i >= 1000) and mc.SAVE_XLA_TIMELINE:
      # produce timeline
      fetched_timeline = timeline.Timeline(run_metadata.step_stats)
      chrome_trace = fetched_timeline.generate_chrome_trace_format()
      with open(os.path.join(mc.TRAIN_DIR,'timeline_02_step_%d.json' % i), 'w') as f:
        f.write(chrome_trace)
      break

  return True



