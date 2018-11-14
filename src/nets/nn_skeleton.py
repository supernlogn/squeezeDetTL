# Author: Ioannis Athanasiadis (ath.ioannis94@gmail.com) 03/14/2017
"""Neural network model base class."""

from __future__ import absolute_import, division, print_function

import numpy as np
from utils import util
import tensorflow as tf
from utils.tf_util import _tf_dense_to_sparse_tensor


def _add_loss_summaries(total_loss):
  """Add summaries for losses
  Generates loss summaries for visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
    model_id: The id of the models inside the model pool.
  """
  losses = tf.get_collection('losses')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    tf.summary.scalar(l.op.name, l)

def _variable_on_device(name, shape, initializer, trainable=True):
  """Helper to create a Variable.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  # TODO(bichen): fix the hard-coded data type below
  dtype = tf.float32
  if not callable(initializer):
    var = tf.get_variable(name, initializer=initializer, trainable=trainable)
  else:
    var = tf.get_variable(
        name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_device(name, shape, initializer, trainable)
  if wd is not None and trainable:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

class ModelSkeleton:
  """Base class of NN detection models."""
  def __init__(self, mc, record_input, base_net=None, global_step=None):
    self.mc = mc

    if global_step != None:
      # self.global_step = tf.Variable(0, name='global_step', trainable=False)
      self.global_step = global_step
    else:
      self.global_step = tf.train.get_or_create_global_step()

    # a scalar tensor in range (0, 1]. Usually set to 0.5 in training phase and
    # 1.0 in evaluation phase
    self.keep_prob = mc.KEEP_PROB if mc.IS_TRAINING else 1.0

    self.image_input = record_input["image/decoded"]
    # A tensor where each element corresponds to a box in an image of a batch of images
    # and its value is the index of the "responsible" anchor box.
    self.aidx = record_input["image/object/bbox/aidx"]
    self.paired_aidx_values = tf.stack([tf.cast(self.aidx.indices[:,0], dtype=tf.int64), self.aidx.values], axis=1)
    # A tensor where an element is 1 if the corresponding box is "responsible"
    # for detection an object and 0 otherwise.
    self.input_mask = tf.scatter_nd(self.paired_aidx_values,
                                    tf.ones_like(self.aidx.values),
                                    [mc.BATCH_SIZE, mc.ANCHORS])

    self.box_input = {
      "image/object/bbox/xmin" : record_input["image/object/bbox/xmin"],
      "image/object/bbox/xmax" : record_input["image/object/bbox/xmax"],
      "image/object/bbox/ymin" : record_input["image/object/bbox/ymin"],
      "image/object/bbox/ymax" : record_input["image/object/bbox/ymax"]}
    self.labels = record_input["image/object/class/label"]
    self.box_delta_input = record_input["image/object/bbox/deltas"]

    if(mc.EVAL_WITH_TRAIN):
      self.filenames = record_input["image/filename"]
      self.widths  = record_input["image/width"]
      self.heights = record_input["image/height"]
  
    # model parameters
    self.model_params = []

    self.viz_op = None

  def _add_forward_graph(self):
    """NN architecture specification."""
    raise NotImplementedError

  def _add_interpretation_graph(self):
    """Interpret NN output."""
    mc = self.mc

    with tf.variable_scope('interpret_output') as scope:
      preds = self.preds
      # probability
      num_class_probs = mc.ANCHOR_PER_GRID*mc.CLASSES
      self.pred_class_probs = tf.reshape(
          tf.nn.softmax(
              tf.reshape(
                  preds[:, :, :, :num_class_probs],
                  [-1, mc.CLASSES])),
          [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
          name='pred_class_probs')
      
      # confidence
      num_confidence_scores = mc.ANCHOR_PER_GRID+num_class_probs
      self.pred_conf = tf.sigmoid(
          tf.reshape(
              preds[:, :, :, num_class_probs:num_confidence_scores],
              [mc.BATCH_SIZE, mc.ANCHORS]),
          name='pred_confidence_score')
      
      # bbox_delta
      self.pred_box_delta = tf.reshape(
          preds[:, :, :, num_confidence_scores:],
          [mc.BATCH_SIZE, mc.ANCHORS, 4],
          name='bbox_delta')
      
      # number of objects. Used to normalize bbox and classification loss
      # self.num_objects = tf.reduce_sum(self.input_mask, name='num_objects')
      self.num_objects = tf.cast(tf.size(self.box_input["image/object/bbox/xmin"].values), tf.float32)

    with tf.variable_scope('bbox') as scope:
      with tf.variable_scope('stretching'):
        delta_x, delta_y, delta_w, delta_h = tf.unstack(
            self.pred_box_delta, axis=2)

        anchor_x = mc.ANCHOR_BOX[:, 0]
        anchor_y = mc.ANCHOR_BOX[:, 1]
        anchor_w = mc.ANCHOR_BOX[:, 2]
        anchor_h = mc.ANCHOR_BOX[:, 3]

        box_center_x = tf.identity(
            anchor_x + delta_x * anchor_w, name='bbox_cx')
        box_center_y = tf.identity(
            anchor_y + delta_y * anchor_h, name='bbox_cy')
        box_width = tf.identity(
            anchor_w * util.safe_exp(delta_w, mc.EXP_THRESH),
            name='bbox_width')
        box_height = tf.identity(
            anchor_h * util.safe_exp(delta_h, mc.EXP_THRESH),
            name='bbox_height')

        self._activation_summary(delta_x, 'delta_x')
        self._activation_summary(delta_y, 'delta_y')
        self._activation_summary(delta_w, 'delta_w')
        self._activation_summary(delta_h, 'delta_h')

        self._activation_summary(box_center_x, 'bbox_cx')
        self._activation_summary(box_center_y, 'bbox_cy')
        self._activation_summary(box_width, 'bbox_width')
        self._activation_summary(box_height, 'bbox_height')
        
      with tf.variable_scope('trimming'):
        xmins, ymins, xmaxs, ymaxs = util.bbox_transform(
            [box_center_x, box_center_y, box_width, box_height])

        # The max x position is mc.IMAGE_WIDTH - 1 since we use zero-based
        # pixels. Same for y.
        xmins = tf.minimum(
            tf.maximum(0.0, xmins), mc.IMAGE_WIDTH-1.0, name='bbox_xmin') # shape = [mc.BATCH_SIZE, mc.ANCHORS]
        self._activation_summary(xmins, 'box_xmin')
        
        ymins = tf.minimum(
            tf.maximum(0.0, ymins), mc.IMAGE_HEIGHT-1.0, name='bbox_ymin') # shape = [mc.BATCH_SIZE, mc.ANCHORS]
        self._activation_summary(ymins, 'box_ymin')

        xmaxs = tf.maximum( 
            tf.minimum(mc.IMAGE_WIDTH-1.0, xmaxs), 0.0, name='bbox_xmax') # shape = [mc.BATCH_SIZE, mc.ANCHORS]
        self._activation_summary(xmaxs, 'box_xmax')

        ymaxs = tf.maximum(
            tf.minimum(mc.IMAGE_HEIGHT-1.0, ymaxs), 0.0, name='bbox_ymax') # shape = [mc.BATCH_SIZE, mc.ANCHORS]
        self._activation_summary(ymaxs, 'box_ymax')

        self.det_boxes = {"xmins": xmins, 
                          "ymins": ymins, 
                          "xmaxs": xmaxs, 
                          "ymaxs": ymaxs}

    with tf.name_scope('IOU'):
      def _tensor_iou(box1, box2):
        with tf.name_scope('intersection'):
          xmin = tf.maximum(box1["xmin"], box2["xmin"], name='xmin')
          ymin = tf.maximum(box1["ymin"], box2["ymin"], name='ymin')
          xmax = tf.minimum(box1["xmax"], box2["xmax"], name='xmax')
          ymax = tf.minimum(box1["ymax"], box2["ymax"], name='ymax')

          w = tf.maximum(0.0, xmax-xmin, name='inter_w')
          h = tf.maximum(0.0, ymax-ymin, name='inter_h')
          intersection = tf.multiply(w, h, name='intersection')

        with tf.name_scope('union'):
          w1 = tf.subtract(box1["xmax"], box1["xmin"], name='w1')
          h1 = tf.subtract(box1["ymax"], box1["ymin"], name='h1')
          w2 = tf.subtract(box2["xmax"], box2["xmin"], name='w2')
          h2 = tf.subtract(box2["ymax"], box2["ymin"], name='h2')

          union = tf.cast(w1*h1 + w2*h2 - intersection, dtype=tf.float32)

        return tf.truediv(tf.cast(intersection, dtype=tf.float32),
                          union + tf.constant(mc.EPSILON, dtype=tf.float32))
      
      mini_ious_values = _tensor_iou(
        {"xmin" : tf.cast(tf.gather_nd(xmins, self.paired_aidx_values), tf.float32), 
         "ymin" : tf.cast(tf.gather_nd(ymins, self.paired_aidx_values), tf.float32), 
         "xmax" : tf.cast(tf.gather_nd(xmaxs, self.paired_aidx_values), tf.float32), 
         "ymax" : tf.cast(tf.gather_nd(ymaxs, self.paired_aidx_values), tf.float32)}, # predicted boxes
        {"xmin" : tf.cast(self.box_input["image/object/bbox/xmin"].values, tf.float32),
         "ymin" : tf.cast(self.box_input["image/object/bbox/ymin"].values, tf.float32),
         "xmax" : tf.cast(self.box_input["image/object/bbox/xmax"].values, tf.float32),
         "ymax" : tf.cast(self.box_input["image/object/bbox/ymax"].values, tf.float32)}) # input boxes
      
      # after computing the ious of the responsible boxes,
      # put the values to a large plane containing all anchors which are responsible and those which are not
      self._ious = tf.scatter_nd(self.paired_aidx_values,
                                 mini_ious_values,
                                 [mc.BATCH_SIZE, mc.ANCHORS])

      self._activation_summary(self._ious, 'conf_score')
      
    with tf.variable_scope('probability') as scope:
      self._activation_summary(self.pred_class_probs, 'class_probs')

      probs = tf.multiply(
          self.pred_class_probs,
          tf.reshape(self.pred_conf, [mc.BATCH_SIZE, mc.ANCHORS, 1]),
          name='final_class_prob')

      self._activation_summary(probs, 'final_class_prob')

      self.det_probs = tf.reduce_max(probs, axis=2, name='score')
      self.det_class = tf.argmax(probs, axis=2, name='class_idx')
      
      self._activation_summary(tf.gather_nd(self.det_class, self.paired_aidx_values), 'detected_classes')

      # get prediction boxes
      self.prediction_boxes, self.score,\
        self.cls_idx_per_img, self.filter_summaries = self.filter_prediction()

  def _add_loss_graph(self):
    """Define the loss operation."""
    mc = self.mc
    
    input_mask = tf.cast(self.input_mask, dtype=tf.float32)

    with tf.variable_scope('class_regression'):
      # cross-entropy: q * -log(p) + (1-q) * -log(1-p)
      # add a small value into log to prevent blowing up
      classes_sparse_indices = tf.concat([self.paired_aidx_values, tf.expand_dims(tf.cast(self.labels.values, dtype=tf.int64),\
                                          axis=1)], axis=1)

      one_hot_labels = tf.scatter_nd(classes_sparse_indices,
                                     tf.reshape(tf.ones_like(self.labels.values, dtype=tf.float32), [-1]),
                                     [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES])
      self.class_loss = tf.truediv(
          tf.reduce_sum(
              (one_hot_labels*(-tf.log(self.pred_class_probs+mc.EPSILON))
               + (1.0-one_hot_labels)*(-tf.log(1-self.pred_class_probs+mc.EPSILON)))
              * tf.expand_dims(input_mask, axis=-1) * mc.LOSS_COEF_CLASS),
          self.num_objects,
          name='class_loss')

      tf.add_to_collection('losses', self.class_loss)

    with tf.variable_scope('confidence_score_regression'):
      self.conf_loss = tf.reduce_mean(
          tf.reduce_sum(
              tf.square((self._ious - self.pred_conf))
              * (input_mask * tf.constant(mc.LOSS_COEF_CONF_POS, tf.float32) / self.num_objects
                 + (1.0 - input_mask) * 
                  tf.constant(mc.LOSS_COEF_CONF_NEG, tf.float32) / 
                  tf.cast(tf.constant(mc.ANCHORS)-tf.cast(self.num_objects, dtype=tf.int32), dtype=tf.float32)),
              reduction_indices=[1]),
          name='confidence_loss')
      tf.add_to_collection('losses', self.conf_loss)
      tf.summary.scalar('mean iou', tf.reduce_sum(self._ious) / self.num_objects)
    
    with tf.variable_scope('bounding_box_regression'):
      delta_x, delta_y, delta_w, delta_h = tf.unstack(
          self.pred_box_delta, axis=2)
      _delta_x = tf.gather_nd(delta_x, self.paired_aidx_values)
      _delta_y = tf.gather_nd(delta_y, self.paired_aidx_values)
      _delta_w = tf.gather_nd(delta_w, self.paired_aidx_values)
      _delta_h = tf.gather_nd(delta_h, self.paired_aidx_values)
      self.bbox_loss = tf.truediv(
          tf.reduce_sum(mc.LOSS_COEF_BBOX * tf.square(tf.stack([
            self.box_delta_input["dx"].values - _delta_x,
            self.box_delta_input["dy"].values - _delta_y,
            self.box_delta_input["dw"].values - _delta_w,
            self.box_delta_input["dh"].values - _delta_h]))),
          self.num_objects,
          name='bbox_loss')

      tf.add_to_collection('losses', self.bbox_loss)

    # add above losses as well as weight decay losses to form the total loss
    self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
  
  def _add_train_graph(self):
    """Define the training operation."""
    mc = self.mc

    lr = tf.train.exponential_decay(mc.LEARNING_RATE,
                                    self.global_step,
                                    mc.DECAY_STEPS,
                                    mc.LR_DECAY_FACTOR,
                                    staircase=True)

    tf.summary.scalar('learning_rate', lr)

    _add_loss_summaries(self.loss)
    if(mc.OPTIMIZER["TYPE"] == "MOMENTUM"):
      opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=mc.OPTIMIZER["MOMENTUM"])
    elif(mc.OPTIMIZER["TYPE"] == "ADAM"):
      opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=mc.OPTIMIZER["BETA1"], beta2=mc.OPTIMIZER["BETA2"])
    grads_vars = opt.compute_gradients(self.loss, 
                                      tf.trainable_variables())

    with tf.variable_scope('clip_gradient') as scope:
      for i, (grad, var) in enumerate(grads_vars):
        grads_vars[i] = (tf.clip_by_norm(grad, mc.MAX_GRAD_NORM), var)

    apply_gradient_op = opt.apply_gradients(grads_vars, global_step=self.global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads_vars:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    with tf.control_dependencies([apply_gradient_op]):
      self.train_op = tf.no_op(name='train')

  def _add_viz_graph(self):
    """Define the visualization operation."""
    mc = self.mc
    with tf.name_scope("viz_graph"):
      # draw ground truth
      bounding_boxes1 = tf.stack([
          tf.cast(tf.sparse_tensor_to_dense(
                  self.box_input["image/object/bbox/ymin"]), tf.float32) / \
                                      tf.constant(mc.IMAGE_HEIGHT, tf.float32),
          tf.cast(tf.sparse_tensor_to_dense(
                  self.box_input["image/object/bbox/xmin"]), tf.float32) / \
                                      tf.constant(mc.IMAGE_WIDTH, tf.float32),
          tf.cast(tf.sparse_tensor_to_dense(
                  self.box_input["image/object/bbox/ymax"]), tf.float32) / \
                                      tf.constant(mc.IMAGE_HEIGHT, tf.float32),
          tf.cast(tf.sparse_tensor_to_dense(
                  self.box_input["image/object/bbox/xmax"]), tf.float32) / \
                                      tf.constant(mc.IMAGE_WIDTH, tf.float32)],
          axis=-1)
      ground_truth_drawn = tf.image.draw_bounding_boxes(
          tf.reverse(
              self.image_input +
              tf.expand_dims(tf.constant(mc.BGR_MEANS,
                                         dtype=tf.float32),
                             axis=0),
              axis=[-1]),
          bounding_boxes1)
      # make prediction boxes dense
      ymins, xmins, ymaxs, xmaxs = [], [], [], []
      _sparse_indices_per_image = []
      _lens = []
      for i in range(mc.BATCH_SIZE):
        ymins.append(self.prediction_boxes[i]["ymins"])
        xmins.append(self.prediction_boxes[i]["xmins"])
        ymaxs.append(self.prediction_boxes[i]["ymaxs"])
        xmaxs.append(self.prediction_boxes[i]["xmaxs"])
        # In order to draw bounding boxes a dense structure is required.
        # To describe the bounding boxes, each of their dimensions should be
        # in the range of [0,1]
        _len = tf.size(ymins[i])
        _lens.append(_len)
        _sparse_indices_per_image.append(
                                  tf.stack([tf.fill(tf.reshape(_len, [1]), i),
                                            tf.range(_len)],
                                            axis=-1))
      _max_len = tf.reduce_max(tf.stack(_lens))
      _sparse_indices = tf.concat(_sparse_indices_per_image, axis=0)
      _ymins = tf.concat(ymins, axis=0) / tf.constant(mc.IMAGE_HEIGHT, tf.float32)
      _xmins = tf.concat(xmins, axis=0) / tf.constant(mc.IMAGE_WIDTH, tf.float32)
      _ymaxs = tf.concat(ymaxs, axis=0) / tf.constant(mc.IMAGE_HEIGHT, tf.float32)
      _xmaxs = tf.concat(xmaxs, axis=0) / tf.constant(mc.IMAGE_WIDTH, tf.float32)
      bounding_boxes2 = tf.stack(
                [tf.sparse_to_dense(_sparse_indices, [mc.BATCH_SIZE, _max_len], _ymins),
                tf.sparse_to_dense(_sparse_indices, [mc.BATCH_SIZE, _max_len], _xmins),
                tf.sparse_to_dense(_sparse_indices, [mc.BATCH_SIZE, _max_len], _ymaxs),
                tf.sparse_to_dense(_sparse_indices, [mc.BATCH_SIZE, _max_len], _xmaxs)], 
                axis=2)
      
      # draw prediction boxes
      self.all_boxes_drawn = tf.image.draw_bounding_boxes(ground_truth_drawn, bounding_boxes2) 

      self.viz_op = tf.summary.image('sample_detection_results',
          self.all_boxes_drawn, collections='image_summary',
          max_outputs=mc.BATCH_SIZE)
      # add summary for the detected classes
      cls_summary = tf.summary.histogram("detected_cls_idx", tf.concat([cls_idx for cls_idx in self.cls_idx_per_img], axis=0), collections='image_summary')
      self.viz_op = tf.summary.merge([self.viz_op, cls_summary])
      # add summary for the ground trouth classes
      cls_summary = tf.summary.histogram("ground_truth_cls_idx", 
                        tf.concat([tf.boolean_mask(self.labels.values, tf.equal(self.labels.indices[:,0], tf.constant(i, dtype=tf.int64))) 
                                   for i in range(mc.BATCH_SIZE)], axis=0), collections='image_summary')
      self.viz_op = tf.summary.merge([self.viz_op, cls_summary, self.filter_summaries])

  def _conv_layer(
      self, layer_name, inputs, filters, size, stride, padding='SAME',
      freeze=False, xavier=False, relu=True, stddev=0.001):
    """Convolutional layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      filters: number of output filters.
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      xavier: whether to use xavier weight initializer or not.
      relu: whether to use relu or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """

    mc = self.mc
    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
        bias_val = cw[layer_name][1]
        # check the shape
        if (kernel_val.shape == 
              (size, size, inputs.get_shape().as_list()[-1], filters)) \
           and (bias_val.shape == (filters, )):
          use_pretrained_param = True
        else:
          print ('Shape of the pretrained parameter of {} does not match, '
              'use randomly initialized parameter'.format(layer_name))
      else:
        print ('Cannot find {} in the pretrained model. Using randomly initialized '
               'parameters'.format(layer_name))

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      channels = inputs.get_shape()[3]

      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val, dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(0.0)
      else:
        if mc.DEBUG_MODE:
          print ('Using randomly initialized parameters for {}'.format(layer_name))        
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(0.0)

      kernel = _variable_with_weight_decay(
          'kernels', shape=[size, size, int(channels), filters],
          wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

      biases = _variable_on_device('biases', [filters], bias_init, 
                                trainable=(not freeze))
      self.model_params += [kernel, biases]

      conv = tf.nn.conv2d(
          inputs, kernel, [1, stride, stride, 1], padding=padding,
          name='convolution')
      conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
  
      if relu:
        if mc.LEAKY_RELU:
          out = tf.nn.leaky_relu(conv_bias, mc.LEAKY_COEF, 'relu')
        else:
          out = tf.nn.relu(conv_bias, 'relu')
      else:
        out = conv_bias

      out_shape = out.get_shape().as_list()
      num_flops = \
        (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]
      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]

      return out
  
  def _pooling_layer(
      self, layer_name, inputs, size, stride, padding='SAME'):
    """Pooling layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
    Returns:
      A pooling layer operation.
    """

    with tf.variable_scope(layer_name) as scope:
      out =  tf.nn.max_pool(inputs, 
                            ksize=[1, size, size, 1], 
                            strides=[1, stride, stride, 1],
                            padding=padding)
      activation_size = np.prod(out.get_shape().as_list()[1:])

      return out

  
  def _fc_layer(
      self, layer_name, inputs, hiddens, flatten=False, relu=True,
      xavier=False, stddev=0.001):
    """Fully connected layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      hiddens: number of (hidden) neurons in this layer.
      flatten: if true, reshape the input 4D tensor of shape 
          (batch, height, weight, channel) into a 2D tensor with shape 
          (batch, -1). This is used when the input to the fully connected layer
          is output of a convolutional layer.
      relu: whether to use relu or not.
      xavier: whether to use xavier weight initializer or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A fully connected layer operation.
    """
    mc = self.mc

    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        use_pretrained_param = True
        kernel_val = cw[layer_name][0]
        bias_val = cw[layer_name][1]

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      input_shape = inputs.get_shape().as_list()
      if flatten:
        dim = input_shape[1]*input_shape[2]*input_shape[3]
        inputs = tf.reshape(inputs, [-1, dim])
        if use_pretrained_param:
          try:
            # check the size before layout transform
            assert kernel_val.shape == (hiddens, dim), \
                'kernel shape error at {}'.format(layer_name)
            kernel_val = np.reshape(
                np.transpose(
                    np.reshape(
                        kernel_val, # O x (C*H*W)
                        (hiddens, input_shape[3], input_shape[1], input_shape[2])
                    ), # O x C x H x W
                    (2, 3, 1, 0)
                ), # H x W x C x O
                (dim, -1)
            ) # (H*W*C) x O
            # check the size after layout transform
            assert kernel_val.shape == (dim, hiddens), \
                'kernel shape error at {}'.format(layer_name)
          except:
            # Do not use pretrained parameter if shape doesn't match
            use_pretrained_param = False
            print ('Shape of the pretrained parameter of {} does not match, '
                   'use randomly initialized parameter'.format(layer_name))
      else:
        dim = input_shape[1]
        if use_pretrained_param:
          try:
            kernel_val = np.transpose(kernel_val, (1,0))
            assert kernel_val.shape == (dim, hiddens), \
                'kernel shape error at {}'.format(layer_name)
          except:
            use_pretrained_param = False
            print ('Shape of the pretrained parameter of {} does not match, '
                   'use randomly initialized parameter'.format(layer_name))

      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val, dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.0)
      else:
        kernel_init = tf.truncated_normal_initializer(
                          stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(0.0)

      weights = _variable_with_weight_decay(
          'weights', shape=[dim, hiddens], wd=mc.WEIGHT_DECAY,
          initializer=kernel_init)
      biases = _variable_on_device('biases', [hiddens], bias_init)
      self.model_params += [weights, biases]
  
      outputs = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
      if relu:
        outputs = tf.nn.relu(outputs, 'relu')

      num_flops = 2 * dim * hiddens + hiddens
      if relu:
        num_flops += 2*hiddens

      return outputs

  def filter_prediction(self):
    """Filter bounding box predictions with probability threshold and
    non-maximum supression.
    Returns:
      final_boxes: 2D array [BATCH_SIZE] of tensors with filtered bounding boxes.
      final_probs: 2D array [BATCH_SIZE] of tensors with filtered probabilities.
      final_cls_idx: 2D array [BATCH_SIZE] of tensors with filtered class indices.
      s: a summary operation
    """
    mc = self.mc
    with tf.name_scope("filter_prediction"):
      ### probability threshold
      # get top N boxes which have greater propability
      with tf.name_scope("probability_threshold"):
        order = tf.contrib.framework.argsort(self.det_probs, axis=1, direction='ASCENDING')[:mc.BATCH_SIZE, :-mc.TOP_N_DETECTION-1:-1]

        # unstack the order array in order to use it for each image separatelly
        unstacked_order = tf.unstack(order, axis=0, num=mc.BATCH_SIZE)
        # for probs
        unstacked_det_probs = tf.unstack(self.det_probs, axis=0, num=mc.BATCH_SIZE)
        unstacked_probs = [tf.gather(unstacked_det_probs[i], unstacked_order[i], axis=0) for i in range(mc.BATCH_SIZE)]

        # for class indicators
        unstacked_cls_idx = tf.unstack(self.det_class, axis=0, num=mc.BATCH_SIZE)
        unstacked_ordered_cls_idx = [tf.gather(unstacked_cls_idx[i], unstacked_order[i], axis=-1) for i in range(mc.BATCH_SIZE)]
        s1 = tf.summary.histogram("unstacked_ordered_cls_idx", tf.concat(unstacked_ordered_cls_idx, axis=0), collections="image_summary")
        # for the boxes
        unstacked_boxes = {tens_name: tf.unstack(self.det_boxes[tens_name], axis=0) for tens_name in self.det_boxes}
        unstacked_ordered_boxes = {tens_name: [tf.gather(unstacked_boxes[tens_name][i],
                                              unstacked_order[i], 
                                              axis=-1) 
                                              for i in range(mc.BATCH_SIZE)] 
                                  for tens_name in unstacked_boxes}
      ### NMS threshold
      with tf.name_scope("NMS_threshold"):
        # get boxes regarding each class and image
        final_boxes_per_img = []
        final_probs_per_img = []
        final_cls_idx_per_img = []
        cls_idx_per_img_before_plot_prob = []
        cls_probs_per_img_before_plot_prob = []
        for i in range(mc.BATCH_SIZE):
          class_mask_per_image = [tf.equal(unstacked_ordered_cls_idx[i], c) 
                                  for c in range(mc.CLASSES)]
          selected_indices_per_cls = []
          ymins_per_cls, xmins_per_cls, ymaxs_per_cls, xmaxs_per_cls = [], [], [], []
          unstacked_probs_per_cls = []
          for c in range(mc.CLASSES):
            class_mask = class_mask_per_image[c]
            ymins_per_cls.append(
                  tf.reshape(tf.boolean_mask(unstacked_ordered_boxes["ymins"][i], class_mask),
                            [-1]))
            xmins_per_cls.append(
                  tf.reshape(tf.boolean_mask(unstacked_ordered_boxes["xmins"][i], class_mask),
                            [-1]))
            ymaxs_per_cls.append(
                  tf.reshape(tf.boolean_mask(unstacked_ordered_boxes["ymaxs"][i], class_mask),
                            [-1]))
            xmaxs_per_cls.append(
                  tf.reshape(tf.boolean_mask(unstacked_ordered_boxes["xmaxs"][i], class_mask),
                            [-1]))
            unstacked_probs_per_cls.append(
                  tf.reshape(tf.boolean_mask(unstacked_probs[i], class_mask),
                            [-1]))
            selected_indices_per_cls.append(
                        tf.reshape(tf.image.non_max_suppression(
                            boxes=tf.stack([ymins_per_cls[c] / tf.constant(mc.IMAGE_HEIGHT, tf.float32),
                                            xmins_per_cls[c] / tf.constant(mc.IMAGE_WIDTH, tf.float32),
                                            ymaxs_per_cls[c] / tf.constant(mc.IMAGE_HEIGHT, tf.float32),
                                            xmaxs_per_cls[c] / tf.constant(mc.IMAGE_WIDTH, tf.float32)],
                                            axis=1),
                            scores=unstacked_probs_per_cls[c],
                            max_output_size=mc.TOP_N_DETECTION,
                            iou_threshold=mc.NMS_THRESH), [-1]))
          _final_boxes_per_img = {
              "ymins": tf.concat([tf.gather(ymins_per_cls[c],
                                  selected_indices_per_cls[c]) for c in range(mc.CLASSES)], axis=0),
              "xmins": tf.concat([tf.gather(xmins_per_cls[c],
                                  selected_indices_per_cls[c]) for c in range(mc.CLASSES)], axis=0),
              "ymaxs": tf.concat([tf.gather(ymaxs_per_cls[c],
                                  selected_indices_per_cls[c]) for c in range(mc.CLASSES)], axis=0),
              "xmaxs": tf.concat([tf.gather(xmaxs_per_cls[c],
                                  selected_indices_per_cls[c]) for c in range(mc.CLASSES)], axis=0)}
          _final_probs_per_img =\
                      tf.concat([tf.gather(unstacked_probs_per_cls[c], selected_indices_per_cls[c])
                                for c in range(mc.CLASSES)], axis=0)
          cls_probs_per_img_before_plot_prob.append([tf.gather(unstacked_probs_per_cls[c], selected_indices_per_cls[c])
                                                     for c in range(mc.CLASSES)])
          _final_cls_idx_per_img =\
                      tf.concat([tf.fill(tf.shape(selected_indices_per_cls[c]), c) for c in range(mc.CLASSES)], axis=0)
          
          cls_idx_per_img_before_plot_prob.append(_final_cls_idx_per_img)
          # filter again using the plotting probability
          # plot_prob_mask = tf.greater(_final_probs_per_img, mc.PLOT_PROB_THRESH)
          final_boxes_per_img.append(_final_boxes_per_img)
          # {
          #     "ymins" : tf.boolean_mask(_final_boxes_per_img["ymins"], plot_prob_mask),
          #     "xmins" : tf.boolean_mask(_final_boxes_per_img["xmins"], plot_prob_mask),
          #     "ymaxs" : tf.boolean_mask(_final_boxes_per_img["ymaxs"], plot_prob_mask),
          #     "xmaxs" : tf.boolean_mask(_final_boxes_per_img["xmaxs"], plot_prob_mask)})

          # final_probs_per_img.append(tf.boolean_mask(_final_probs_per_img, plot_prob_mask))
          final_probs_per_img.append(_final_probs_per_img)
          # final_cls_idx_per_img.append(tf.boolean_mask(_final_cls_idx_per_img, plot_prob_mask))
          final_cls_idx_per_img.append(_final_cls_idx_per_img)
        s2 = tf.summary.histogram("cls_idx_per_img_before_plot_prob", tf.concat(cls_idx_per_img_before_plot_prob, axis=0), collections="image_summary")  
        s3 = [tf.summary.histogram("cls_probs_bef_p_p_0", tf.concat([sc[0] for sc in cls_probs_per_img_before_plot_prob], axis=0)), 
              tf.summary.histogram("cls_probs_bef_p_p_1", tf.concat([sc[1] for sc in cls_probs_per_img_before_plot_prob], axis=0)),
              tf.summary.histogram("cls_probs_bef_p_p_2", tf.concat([sc[2] for sc in cls_probs_per_img_before_plot_prob], axis=0))]

    return final_boxes_per_img, final_probs_per_img, final_cls_idx_per_img, tf.summary.merge([s1,s2])

  def _activation_summary(self, x, layer_name):
    """Helper to create summaries for activations.

    Args:
      x: layer output tensor
      layer_name: name of the layer
    Returns:
      nothing
    """
    with tf.variable_scope('activation_summary'):
      tf.summary.histogram(
          'activation_summary/'+layer_name, x)
      tf.summary.scalar(
          'activation_summary/'+layer_name+'/sparsity', tf.nn.zero_fraction(x))
      tf.summary.scalar(
          'activation_summary/'+layer_name+'/average', tf.reduce_mean(x))
      tf.summary.scalar(
          'activation_summary/'+layer_name+'/max', tf.reduce_max(x))
      tf.summary.scalar(
          'activation_summary/'+layer_name+'/min', tf.reduce_min(x))

  def geteval_op_list(self):
    """get all tensorflow operations regarding this 
       model evaluation.
    """
    detection_boxes =  tf.transpose(
            tf.stack(util.bbox_transform_inv([self.det_boxes["xmins"], self.det_boxes["ymins"], self.det_boxes["xmaxs"], self.det_boxes["ymaxs"]])),
            (1, 2, 0), name='bbox'
        )
    
    return filter(lambda x: x != None, 
            [self.prediction_boxes, self.score, self.cls_idx_per_img,
              self.filenames, self.widths, self.heights, self.viz_op, self.det_boxes, self.det_probs, self.det_class])
