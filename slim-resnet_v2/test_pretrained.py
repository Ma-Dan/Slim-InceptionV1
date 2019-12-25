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
from tensorflow.contrib.slim.python.slim.nets import resnet_v2

def test_network(img_path, label_path):
    x = tf.placeholder("float", shape=[None, 224, 224, 3], name='input')
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, end_points = resnet_v2.resnet_v2_50(x, num_classes=1001, is_training=False)
    predictions = end_points["predictions"]
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "resnet_v2_50.ckpt");

        imgfloat = tf.cast(tf.image.decode_jpeg(tf.read_file(img_path), channels=3), dtype=tf.float32)
        img = tf.subtract(tf.multiply(tf.div(tf.image.resize_images(tf.expand_dims(imgfloat, 0), (224, 224), method=0), 255.0), 2), 1.0)
        predictions_val = predictions.eval(feed_dict={x: img.eval()})
        predicted_classes = np.argmax(predictions_val, axis=3)

        file = open(label_path, encoding="utf-8")
        labels = file.readlines()
        print(predicted_classes, labels[predicted_classes[0][0][0]])

def main():
    test_network(sys.argv[1], "labels.txt")

main()

