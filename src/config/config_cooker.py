# Author Ioannis Athanasiadis (ath.ioannis94@gmail.com)

"""Arbitary Model configurations"""

import os
import json
import numpy as np
from easydict import EasyDict as edict
from net_shape_calculator import _get_output_shape

__all__ = ["base_model_config",
           "cook_config",
           "set_anchors",
           "direct_cook_config"]

def base_model_config():
  """ This creates the basic model, which needs only few adjustements by the other configurations,
      in order to work for other datasets and/or detection neural networks.
  """
  BASE_CONFIG_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", "base_config.json")
  
  with open(BASE_CONFIG_FILENAME, 'r') as fp:
    cfg = edict(json.load(fp, encoding="utf8"))

  return cfg

def cook_config(ext_config_filename):
  """
    This function is usefull for merging basic config with external configs.
    External config's constants overwrite those of basic config.
    Args:
        ext_config_filename: the path for the external config json file. 
  """
  mc = base_model_config()
  with open(ext_config_filename, "r") as fp:
    ext_mc = edict(json.load(fp, encoding="utf8"))
    for s in ext_mc.keys():
      mc[s] = ext_mc[s]
  # mc.ANCHOR_BOX            = set_anchors(mc)
  # print(np.max(np.square(np.array(set_anchors_testing(mc)) - np.array(set_anchors(mc)))))
  # mc.ANCHORS               = len(mc.ANCHOR_BOX)
  # H, W, C                  = _get_output_shape(mc)
  # mc.MODEL_OUTPUT_SHAPE    = [H, W, mc.ANCHOR_PER_GRID]
  return mc

def direct_cook_config(nmc, do_set_anchors=True):
  mc = base_model_config()
  # override and register new members to dict  
  for s in nmc:
    mc[s] = nmc[s]
  # if(do_set_anchors):
  #   mc.ANCHOR_BOX            = set_anchors(mc)
  #   mc.ANCHORS               = len(mc.ANCHOR_BOX)
  # H, W, C                  = _get_output_shape(mc)
  # mc.MODEL_OUTPUT_SHAPE    = [H, W, mc.ANCHOR_PER_GRID]
  return mc

def set_anchors(mc):
  """
    This function returns the default anchors given the image shapes and the
    anchors per grid point. The grid has width and height equal to the final's 
    layer output.
    Args:
      mc model configuration
    returns:
      default anchors
  """
  H, W, C = _get_output_shape(mc)
  B = mc.ANCHOR_PER_GRID
  X = np.array(mc.INITIAL_ANCHOR_SHAPES)
  X[:,0] *= mc.IMAGE_WIDTH
  X[:,1] *= mc.IMAGE_HEIGHT
  anchor_shapes = np.reshape( # it refers to the anchor width and height
      [X] * H * W,
      (H, W, B, 2)
  )
  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(mc.IMAGE_WIDTH)/(W+1)]*H*B), 
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(mc.IMAGE_HEIGHT)/(H+1)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )

  return anchors

def set_anchors_testing(mc):
  H, W, B = 24, 78, 9
  anchor_shapes = np.reshape(
      [np.array(
          [[  36.,  37.], [ 366., 174.], [ 115.,  59.],
           [ 162.,  87.], [  38.,  90.], [ 258., 173.],
           [ 224., 108.], [  78., 170.], [  72.,  43.]])] * H * W,
      (H, W, B, 2)
  )
  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(mc.IMAGE_WIDTH)/(W+1)]*H*B), 
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(mc.IMAGE_HEIGHT)/(H+1)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )

  return anchors
