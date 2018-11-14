# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import joblib
import tensorflow as tf

from nets.nn_skeleton import ModelSkeleton

class SqueezeDet(ModelSkeleton):
  def __init__(self, mc, record_input, gpu_id=0, base_net=None, global_step=None):
    with tf.device('/job:localhost/replica:0/task:0/device:CPU:0'):
      ModelSkeleton.__init__(self, mc, record_input, base_net=base_net, global_step=global_step)

      self._add_forward_graph()
      self._add_interpretation_graph()
      self._add_loss_graph()
      if mc.IS_TRAINING:
        self._add_train_graph()
      if mc.VISUALIZE_ON:
        self._add_viz_graph()

  def _add_forward_graph(self):
    """NN architecture."""

    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)

      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

    self.conv1 = self._conv_layer(
        'conv1', self.image_input, filters=64, size=3, stride=2,
        padding='SAME', freeze=mc.FREEZE_LAYERS["conv1"])
    self.pool1 = self._pooling_layer(
        'pool1', self.conv1, size=3, stride=2, padding='SAME')

    self.fire2 = self._fire_layer(
        'fire2', self.pool1, s1x1=16, e1x1=64, e3x3=64, freeze=mc.FREEZE_LAYERS["fire2"])
    self.fire3 = self._fire_layer(
        'fire3', self.fire2, s1x1=16, e1x1=64, e3x3=64, freeze=mc.FREEZE_LAYERS["fire3"])
    self.pool3 = self._pooling_layer(
        'pool3', self.fire3, size=3, stride=2, padding='SAME')

    self.fire4 = self._fire_layer(
        'fire4', self.pool3, s1x1=32, e1x1=128, e3x3=128, freeze=mc.FREEZE_LAYERS["fire4"])
    self.fire5 = self._fire_layer(
        'fire5', self.fire4, s1x1=32, e1x1=128, e3x3=128, freeze=mc.FREEZE_LAYERS["fire5"])
    self.pool5 = self._pooling_layer(
        'pool5', self.fire5, size=3, stride=2, padding='SAME')
    
    self.fire6 = self._fire_layer(
        'fire6', self.pool5, s1x1=48, e1x1=192, e3x3=192, freeze=mc.FREEZE_LAYERS["fire6"])
    self.fire7 = self._fire_layer(
        'fire7', self.fire6, s1x1=48, e1x1=192, e3x3=192, freeze=mc.FREEZE_LAYERS["fire7"])
    self.fire8 = self._fire_layer(
        'fire8', self.fire7, s1x1=64, e1x1=256, e3x3=256, freeze=mc.FREEZE_LAYERS["fire8"])
    self.fire9 = self._fire_layer(
        'fire9', self.fire8, s1x1=64, e1x1=256, e3x3=256, freeze=mc.FREEZE_LAYERS["fire9"])

    # Two extra fire modules that are not trained before
    self.fire10 = self._fire_layer(
        'fire10', self.fire9, s1x1=96, e1x1=384, e3x3=384, freeze=mc.FREEZE_LAYERS["fire10"])
    self.fire11 = self._fire_layer(
        'fire11', self.fire10, s1x1=96, e1x1=384, e3x3=384, freeze=mc.FREEZE_LAYERS["fire11"])
    self.dropout11 = tf.nn.dropout(self.fire11, self.keep_prob, name='drop11')

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    self.preds = self._conv_layer(
        'conv12', self.dropout11, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001, freeze=mc.FREEZE_LAYERS["conv12"])

  def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.01,
      freeze=False):
    """Fire layer constructor.

    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """

    sq1x1 = self._conv_layer(
        layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex1x1 = self._conv_layer(
        layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex3x3 = self._conv_layer(
        layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)

    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')
