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

slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets import inception_v1
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow

def ckpt_info(ckpt_file):
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        print(reader.get_tensor(key))

def test_network(img_path, label_path):
    x = tf.placeholder("float", shape=[None, 224, 224, 3], name='input')
    xscale = tf.subtract(tf.multiply(tf.div(x, 255.0), 2), 1.0)
    with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
        logits, end_points = inception_v1.inception_v1(xscale, num_classes=1001, dropout_keep_prob=1.0, is_training=False)
    predictions = tf.nn.softmax(logits, name="output")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "inception_v1.ckpt")
        #ckpt_info("inception_v1.ckpt")

        #var_list = tf.global_variables()
        #print(var_list)
        constant_graph = tf.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            constant_graph,
            ['output']
        )
        with tf.gfile.GFile("inception_v1.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())

        imgfloat = tf.cast(tf.image.decode_jpeg(tf.read_file(img_path), channels=3), dtype=tf.float32)
        img = tf.image.resize_images(tf.expand_dims(imgfloat, 0), (224, 224), method=0)
        predictions_val = predictions.eval(feed_dict={x: img.eval()})
        predicted_classes = np.argmax(predictions_val, axis=1)

        file = open(label_path)
        labels = file.readlines()
        print(predicted_classes, labels[predicted_classes[0]])

def main():
    test_network(sys.argv[1], "labels.txt")

main()

