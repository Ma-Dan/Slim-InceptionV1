# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition for inception v1 classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
import os
import glob
from skimage import io, transform
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import control_flow_ops

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

import collections
import time

slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v2

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('checkpointDir', './model', 'modeloutput')

def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.9,
                     batch_norm_epsilon=1e-3,
                     batch_norm_scale=True,
                     activation_fn=tf.nn.relu,
                     use_batch_norm=True,
                     batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
                     batch_norm_var_collection='moving_vars'):
  """Defines the default ResNet arg scope.
  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    activation_fn: The activation function which is used in ResNet.
    use_batch_norm: Whether or not to use batch normalization.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.
  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': batch_norm_updates_collections,
      'fused': None,  # Use fused batch norm if possible.
      # collection containing the moving mean and moving variance.
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      }
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=activation_fn,
      normalizer_fn=slim.batch_norm if use_batch_norm else None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc


def trainmodel(train_batch, train_label_batch, val_label_batch, num_epochs):
    with slim.arg_scope(resnet_arg_scope()):
        train_logits, end_points = resnet_v2.resnet_v2_50(train_batch, num_classes=2, is_training=True)

    tf.losses.sparse_softmax_cross_entropy(labels=train_label_batch, logits=train_logits)
    total_loss = tf.losses.get_total_loss()
    global_step = tf.Variable(0, name='global_step', trainable=False)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(total_loss, global_step=global_step)

    prediction_labels = tf.argmax(end_points['predictions'], 3)
    correct_prediction = tf.equal(prediction_labels, val_label_batch)
    train_accuracy_batch = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver(tf.trainable_variables() + tf.get_collection_ref("moving_vars"))

    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        print("Initialized!")

        step = 0
        start_time = time.time()
        for epoch_index in range(num_epochs):
            _, loss_out, train_acc_out = sess.run([train_op, total_loss, train_accuracy_batch])

            duration = time.time() - start_time
            start_time = time.time()

            print("Minibatch loss at step %d: %.6f (%.3f sec)" % (step, loss_out, duration))
            print("Minibatch accuracy: %.6f" % train_acc_out)

            step += 1

        print("Saving checkpoint...")
        saver.save(sess, './train.ckpt')
        print("Checkpoint saved!")


def main(_):
    path = './picture/'
    w = 224
    h = 224
    c = 3
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    train_labels = []
    val_labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading training image: %s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)
            train_labels.append(0 if idx == 0 else 1)
            val_labels.append(0 if idx == 0 else 1)

    data = np.asarray(imgs, np.float32)
    train_labels = np.asarray(train_labels, np.int64)
    val_labels = np.asarray(val_labels, np.int64)

    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    train_labels = train_labels[arr]
    val_labels = val_labels[arr]

    train_labels = np.reshape(train_labels, (train_labels.shape[0], 1, 1, 1))
    val_labels = np.reshape(val_labels, (val_labels.shape[0], 1, 1))

    x_train = data
    y_train = tf.cast(tf.constant(train_labels), dtype=tf.int32)

    trainmodel(x_train, y_train, tf.constant(val_labels), 100)

if __name__ == "__main__":
    tf.app.run()
