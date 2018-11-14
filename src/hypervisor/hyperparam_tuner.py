import json
import os
import sys

import ast
import numpy as np
from lipo_agent import adalipo_search
from random_agent import random_search
from easydict import EasyDict as edict
from dlib import function_evaluation as function_evaluation_obj

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from supervisor.trainer import train as sup_train


def np_array(*args):
  """ np.array wrapper
  """
  return np.array(args)

hfuncs = {
  "logspace": np.logspace,
  "arange": np.arange,
  "linspace": np.linspace,
  "array" : np_array
}


def parse_mc_option(mc_option, hfuncs):
  """ parses a string of model config hyperparameter value
      into an array.
      Returns:
        1) list of values as the user requested
        2) if the list accepts integer values only
  """
  if type(mc_option) == type(edict()):
    if "HVALUE" in mc_option:
      str_func, str_args = mc_option["HVALUE"].split("(")
      args = [float(str_arg) for str_arg in str_args[:-1].split(",")]
      if str_func in hfuncs:
        val_list = hfuncs[str_func] (*args)
        if "IS_INTEGER" in mc_option:
          return val_list, mc_option["IS_INTEGER"]
        else:
          return val_list, False
      else:
        print("ERROR NO SUCH HFUNC")
        return None
    else:
        return mc_option, None
  else:
    return mc_option, None
  return mc_option, None

def parse_mc(mc):
  """
    Parses all mc to find hopt vars to hyperoptimize and to edit
    in every hyperoptimization iteration.
    Args:
      mc: model configuration
    Returns:
      mc: new model_configuration
      hopt_vars: array of triples (mc key, array of possible values of mc[key], is_integer)
  """
  hopt_vars = []
  for x in mc.keys():
    val_ar, r = parse_mc_option(mc[x], hfuncs)
    if r != None:
      hopt_vars.append((x, val_ar, r))
    else:
      mc[x] = val_ar
  new_mc = edict(mc.copy())

  return new_mc, hopt_vars

def create_mc_from_hopt_vars(mc, hopt_vars_vals, hopt_vars_keys, is_integer):
  """
  Args:
    mc: basic model configuration
    hopt_vars_vals: Array of hyperoptimization values proposal
    hopt_vars_keys: keys of the returning model config
    is_integer: Array indicating if corresponding variable in hopt_vars_vals is integer
  Returns:
    rmc: a proper dictionary for model configuration   
  """
  rmc = edict(mc.copy())
  i = 0
  for key, val in zip(hopt_vars_keys, hopt_vars_vals):    
    if is_integer[i]:
      rmc[key] = int(val)
    else:
      rmc[key] = val
    i = i + 1
  
  return edict(rmc)

def train_and_evaluate(mc):
  """
    start training and evaluate it after the last step
    Args:
      mc: model configuration dictionary
    Returns:
      The maximum mAP found in the total training process.
  """
  # start training
  sup_train(mc)

  # read evaluation map file if exists
  eval_file_path = os.path.join(mc["EVAL_DIR"], "mAP_history.json")
  with open(eval_file_path, "r") as f:
    # mAP_per_step = f.readlines()
    # mAPs = [float(l.split("\"mAP\": ")[-1]) for l in mAP_per_step if "mAP" in l]
    mAPs = [h["mAP"] for h in json.load(f)["history"]]
    max_mAP = np.max(mAPs)

  return max_mAP

def create_trainer(mc, hopt_vars, is_integer):
  """
    Creates a trainer function to be used by the dlib hyperoptimization functions
    Args:
      mc: basic model configuration
      hopt_vars: array of pairs (mc key, array of possible values of mc[key])
      is_integer: Array indicating if corresponding variable in hopt_vars_vals is integer
  """
  hopt_vars_keys = [p[0] for p in hopt_vars]

  def _train(*args):
    xmc = create_mc_from_hopt_vars(mc, args, hopt_vars_keys, is_integer)
    hyperiteration_string =  '_'.join([(str(k) + "_" + str(v)) for k,v in zip(hopt_vars_keys, args)])
    new_TRAIN_DIR = os.path.join(mc["BASE_DIR"], "train" + hyperiteration_string)
    new_eval_dir = os.path.join(mc["BASE_DIR"], "eval" + hyperiteration_string)

    if not os.path.exists(new_TRAIN_DIR):
      os.mkdir(new_TRAIN_DIR)
    xmc["TRAIN_DIR"] = new_TRAIN_DIR
    xmc.TRAIN_DIR = new_TRAIN_DIR

    if not os.path.exists(new_eval_dir):
      os.mkdir(new_eval_dir)
    xmc["EVAL_DIR"] = new_eval_dir
    xmc.EVAL_DIR = new_eval_dir

    return train_and_evaluate(xmc)
  
  return _train

def hopt_training(mc):
  """
    Starts the hyperoptimization training based on mc values
    Args:
      mc: model configuration as given by the user
  """
  mc, hopt_vars = parse_mc(mc)
  hopt_vars_keys = [p[0] for p in hopt_vars]
  
  mins = [p[1][0] for p in hopt_vars]
  maxs = [p[1][-1] for p in hopt_vars]
  is_integer = [p[2] for p in hopt_vars]

  # read history (if any)
  hist_path = os.path.join(mc["BASE_DIR"], "hp_opt_history.txt")
  initial_values = []
  if os.path.exists(hist_path):
    with open(hist_path, "r") as f:
      hist_strings = f.read()
    for h in hist_strings:
      t = [ast.literal_eval(x) for x in h[1:-1].split(",")]
      x = t[:-1]
      func_val = t[-1]
      initial_values.append([function_evaluation_obj(x, func_val)])

  # optimize
  opt_hp, opt_eval = adalipo_search(mc, create_trainer(mc, hopt_vars, is_integer), mc["HOPT"]["MAX_ITERATIONS"], 
                                    mins, maxs, is_integer, initial_values=initial_values)

  # report best result
  with open(os.path.join(mc["BASE_DIR"], "hopt_results.txt"), "w") as f:
    f.write("Best evaluation is: %f\n" % float(opt_eval))
    f.write("optimal variables are:\n")
    for k,v in zip(hopt_vars_keys, opt_hp):
      f.write("%s:  %f\n" %(k, v))
  
  return True
