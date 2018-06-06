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
from tensorflow.contrib.slim.python.slim.nets import inception_v1

tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.flags.FLAGS

def inception_v1_arg_scope(weight_decay=0.00004,
                           use_batch_norm=True,
                           batch_norm_var_collection='moving_vars'):
  """Defines the default InceptionV1 arg scope.

  Note: Althougth the original paper didn't use batch_norm we found it useful.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    batch_norm_var_collection: The name of the collection for the batch norm
      variables.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': 0.9,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # collection containing update_ops.
      'updates_collections': ops.GraphKeys.UPDATE_OPS,
      # collection containing the moving mean and moving variance.
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      }
  }
  if use_batch_norm:
    normalizer_fn = layers_lib.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with arg_scope(
      [layers.conv2d, layers_lib.fully_connected],
      weights_regularizer=regularizers.l2_regularizer(weight_decay)):
    with arg_scope(
        [layers.conv2d],
        weights_initializer=initializers.variance_scaling_initializer(),
        activation_fn=nn_ops.relu,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc


def trainmodel(train_batch, train_label_batch, train_label, num_epochs):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index,
                             config = config)
    print("Cluster job: %s, task_index: %d, target: %s" % (FLAGS.job_name, FLAGS.task_index, server.target))
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
            with slim.arg_scope(inception_v1_arg_scope()):
                train_logits, end_points = inception_v1.inception_v1(train_batch, num_classes=2, is_training=True)

            tf.losses.sparse_softmax_cross_entropy(labels=train_label, logits=train_logits)
            total_loss = tf.losses.get_total_loss()
            global_step = tf.Variable(0, name='global_step', trainable=False)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
                train_op = optimizer.minimize(total_loss, global_step=global_step)

            prediction_labels = tf.argmax(end_points['Predictions'], 1)
            read_labels = tf.argmax(train_label_batch, 1)
            correct_prediction = tf.equal(prediction_labels, read_labels)
            train_accuracy_batch = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            saver = tf.train.Saver(tf.trainable_variables() + tf.get_collection_ref("moving_vars"))
            #init_op = tf.global_variables_initializer()
            local_init_op = tf.global_variables_initializer()

        # Create a "Supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="./tflog",
                                 #init_op=init_op,
                                 local_init_op=local_init_op,
                                 saver = saver,
                                 global_step=global_step,
                                 save_model_secs=600)

        # The supervisor takes care of session initialization and restoring from
        # a checkpoint.
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        #session = tf.Session(config=config, ...)
        sess = sv.prepare_or_wait_for_session(server.target, config = config)

        # Start queue runners for the input pipelines (if ang).
        sv.start_queue_runners(sess)

        print("Initialized")

        step = 0
        start_time = time.time()
        for epoch_index in range(num_epochs):
            _, l, end_points2, logits2, train_acc2_batch = sess.run([train_op, total_loss, end_points, train_logits, train_accuracy_batch])

            duration = time.time() - start_time

            print("Minibatch loss at step %d: %.6f (%.3f sec)" % (step, l, duration))
            print(end_points2['Predictions'])
            print("Minibatch accuracy: %.6f" % train_acc2_batch)
            #print("lr: %.6f" % optimizer._lr)

            step += 1

        sv.stop()


def main(_):
    path = './picture/'
    w = 224
    h = 224
    c = 3
    cate   = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs   = []
    labels = []
    train_labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the image: %s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)
            labels.append([1, 0] if idx == 0 else [0, 1])
            train_labels.append(0 if idx == 0 else 1)
            #break
        #break

    data = np.asarray(imgs, np.float32)
    label = np.asarray(labels, np.int32)
    train_label = np.asarray(train_labels, np.int32)
    
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]
    train_label = train_label[arr]
    
    x_train = data
    y_train = tf.cast(tf.constant(label), dtype=tf.float32)
    
    trainmodel(x_train, y_train, tf.constant(train_label), 100)
    
if __name__ == "__main__":
    tf.app.run()
