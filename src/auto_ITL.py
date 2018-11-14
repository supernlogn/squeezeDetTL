#!/usr/bin/python
import os
import numpy as np
import tensorflow as tf

from config.config_cooker import *
from supervisor.trainer import train
from hypervisor import hyperparam_tuner
from supervisor.evaluator import evaluate_once
from supervisor.load_config import super_config

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config_file', '', """Path to json configuration file""")

def start(args):  # pylint: disable=unused-argument
  """
    Descr:
      initiates the training process, to the right path with the right configuration
      file.
    Args:
      args: a dictionary containing
        ["TRAIN_DIR"] = Where checkpoints and tensorboard files will be saved during training
        ["ext_config_file_path"] = A string of the external configuration json file path
        ["mc_direct"] = whether to read the model configuration file directly
  """
  sc = super_config()
  if("mc_direct" in args and args["mc_direct"]):
    mc = direct_cook_config(args, do_set_anchors=True)
  else:
    mc = cook_config(args["ext_config_file_path"])
  mc.ANCHOR_BOX = np.array(mc.ANCHOR_BOX)
  # copy non existent key-value pairs from sc to mc
  for c in sc:
    if(not c in mc.keys()):
      mc[c] = sc[c]

  # make a train dir target
  if not "TRAIN_DIR" in mc:
    mc["TRAIN_DIR"] = os.path.join(mc["BASE_DIR"], "train")

  # make an eval dir target
  if not "EVAL_DIR" in mc:
    mc["EVAL_DIR"] = os.path.join(mc["BASE_DIR"], "evals")

  os.environ['CUDA_VISIBLE_DEVICES'] = str(mc["GPU"])
  if(mc["IS_TRAINING"]):
    if("HOPT" in mc):
      hyperparam_tuner.hopt_training(mc)
    else:
      train(mc)
  else:
    evaluate_once(mc)
  return

def main(_):
  """
    check arguments provided by the user and launch
    the framework.
  """
  assert FLAGS.config_file != '',\
          'No config file was specified'
  assert os.path.exists(FLAGS.config_file),\
          'Config file path does not exist'
  
  start(args={"ext_config_file_path": FLAGS.config_file})
  
  return

if __name__ == "__main__":
  tf.app.run()