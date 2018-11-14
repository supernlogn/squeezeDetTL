""" Calculate the shape of a net's output's based on
    its configuration files.
    author: Ioannis Athansiadis (supernlogn)
"""
import numpy as np

__all__ = ["_get_output_shape"]

def _get_output_shape(mc):
  """ Given the image shape from the model configuration,
      this function produces the final feature output shape
      @param mc model configuration dictionary
      @Note channel num is not always right
  """  

  # # # shapes are (H, W, C) using the standard tensorflow paradigm
  if(mc.NET == "squeezeDet+"):
    ret_shape = get_squeezeDetPlus_shape(mc)
  elif(mc.NET == "squeezeDet"):
    ret_shape = get_squeezeDet_shape(mc)
  else:
    ret_shape = None

  return (int(ret_shape[0]), int(ret_shape[1]), int(ret_shape[2]))


def get_conv2d_out_shape(in_dims, filter_sizes, strides, padding):
  """ @param in_dims = input (height, width, channels)
      @param filter_sizes = (kernel height, kernel width, number of filters)
      @param strides = (filter height stride, filter width stride)
  """
  if(padding == 'SAME'):
    return (np.ceil(float(in_dims[0]) / float(strides[0])),
            np.ceil(float(in_dims[1]) / float(strides[1])),
            float(filter_sizes[2]))
  elif(padding == 'VALID'):
    return (np.ceil(float(in_dims[0] - filter_sizes[0] + 1) / float(strides[0])),
            np.ceil(float(in_dims[1] - filter_sizes[1] + 1) / float(strides[1])),
            float(filter_sizes[2]))

def get_pool_out_shape(in_dims, filter_sizes, strides, padding):
  if(padding == 'SAME'):
    return (np.ceil(float(in_dims[0]) / float(strides[0])),
            np.ceil(float(in_dims[1]) / float(strides[1])),
            in_dims[2])
  elif(padding == 'VALID'):
    return (np.ceil(float(in_dims[0] - filter_sizes[0] + 1) / float(strides[0])),
            np.ceil(float(in_dims[1] - filter_sizes[1] + 1) / float(strides[1])),
            in_dims[2])

def get_fire_out_shape(in_dims, s1x1, e1x1, e3x3):
  sq1x1_shape = get_conv2d_out_shape(in_dims, (1, 1, s1x1), [1, 1],'SAME')
  ex1x1_shape = get_conv2d_out_shape(sq1x1_shape, (1, 1, e1x1), [1, 1], 'SAME')
  ex3x3_shape = get_conv2d_out_shape(sq1x1_shape, (3, 3, e3x3), [1, 1], 'SAME')
  return (ex3x3_shape[0], ex3x3_shape[1], ex3x3_shape[2] + ex1x1_shape[2]) 

def get_squeezeDetPlus_shape(mc):
  input_shape = (mc.IMAGE_HEIGHT,
                mc.IMAGE_WIDTH,
                3)
  conv1_shape = get_conv2d_out_shape(input_shape, 
                                    (7, 7, 96),
                                    [2, 2], 'VALID')
  pool1_shape = get_pool_out_shape(conv1_shape,
                                  (3, 3), 
                                  [2, 2], 'VALID')
  fire2_shape = get_fire_out_shape(pool1_shape,
                                  s1x1=96, e1x1=64,
                                  e3x3=64)
  fire3_shape = get_fire_out_shape(fire2_shape,
                                  s1x1=96, e1x1=64,
                                  e3x3=64)
  fire4_shape = get_fire_out_shape(fire3_shape,
                                  s1x1=192, e1x1=128,
                                  e3x3=128)
  pool4_shape = get_pool_out_shape(fire4_shape,
                                  (3, 3),
                                  [2, 2], 'VALID')
  fire5_shape = get_fire_out_shape(pool4_shape,
                                  s1x1=192, e1x1=128,
                                  e3x3=128)
  fire6_shape = get_fire_out_shape(fire5_shape,
                                  s1x1=288, e1x1=192,
                                  e3x3=192)
  fire7_shape = get_fire_out_shape(fire6_shape,
                                  s1x1=288, e1x1=192,
                                  e3x3=192)
  fire8_shape = get_fire_out_shape(fire7_shape,
                                  s1x1=384, e1x1=256,
                                  e3x3=256)
  pool8_shape = get_pool_out_shape(fire8_shape,
                                  (3, 3),
                                  [2, 2], 'VALID')
  fire9_shape = get_fire_out_shape(pool8_shape,
                                  s1x1=384, e1x1=256,
                                  e3x3=256)
  fire10_shape = get_fire_out_shape(fire9_shape,
                                    s1x1=384, e1x1=256,
                                    e3x3=256)
  fire11_shape = get_fire_out_shape(fire10_shape,
                                    s1x1=384, e1x1=256,
                                    e3x3=256)
  num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
  conv12_shape = get_conv2d_out_shape(fire11_shape,
                                      (3, 3, num_output), 
                                      [1, 1], 'SAME')
  return conv12_shape


def get_squeezeDet_shape(mc):
  input_shape = (mc.IMAGE_HEIGHT,
                mc.IMAGE_WIDTH,
                3)
  conv1_shape = get_conv2d_out_shape(input_shape, 
                                    (3, 3, 64),
                                    [2, 2], 'SAME')
  pool1_shape = get_pool_out_shape(conv1_shape,
                                  (3, 3), 
                                  [2, 2], 'SAME')
  fire2_shape = get_fire_out_shape(pool1_shape,
                                  s1x1=16, e1x1=64,
                                  e3x3=64)
  fire3_shape = get_fire_out_shape(fire2_shape,
                                  s1x1=16, e1x1=64,
                                  e3x3=64)
  pool3_shape = get_pool_out_shape(fire3_shape,
                                   (3, 3),
                                   [2, 2], 'SAME')
  fire4_shape = get_fire_out_shape(pool3_shape,
                                  s1x1=32, e1x1=128,
                                  e3x3=128)
  fire5_shape = get_fire_out_shape(fire4_shape,
                                  s1x1=32, e1x1=128,
                                  e3x3=128)
  pool5_shape = get_pool_out_shape(fire5_shape,
                                   (3, 3),
                                   [2, 2], 'SAME')
  fire6_shape = get_fire_out_shape(pool5_shape,
                                  s1x1=48, e1x1=192,
                                  e3x3=192)
  fire7_shape = get_fire_out_shape(fire6_shape,
                                  s1x1=48, e1x1=192,
                                  e3x3=192)
  fire8_shape = get_fire_out_shape(fire7_shape,
                                  s1x1=64, e1x1=256,
                                  e3x3=256)
  fire9_shape = get_fire_out_shape(fire8_shape,
                                  s1x1=64, e1x1=256,
                                  e3x3=256)
  fire10_shape = get_fire_out_shape(fire9_shape,
                                    s1x1=96, e1x1=384,
                                    e3x3=384)
  fire11_shape = get_fire_out_shape(fire10_shape,
                                    s1x1=96, e1x1=384,
                                    e3x3=384)
  num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
  conv12_shape = get_conv2d_out_shape(fire11_shape,
                                      (3, 3, num_output), 
                                      [1, 1], 'SAME')
  return conv12_shape