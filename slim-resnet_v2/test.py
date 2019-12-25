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
from tensorflow.python.framework import graph_util
import sys
from skimage import io, transform


slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v2

def test_network(img_path):
    x = tf.placeholder("float", shape=[None, 224, 224, 3], name='input')
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, end_points = resnet_v2.resnet_v2_50(x, num_classes=2, is_training=False)
    predictions = end_points["predictions"]

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "train.ckpt")

        w = 224
        h = 224
        c = 3
        imgs = []
        img = io.imread(img_path)
        img = transform.resize(img, (w, h, c))
        imgs.append(img)

        data = np.asarray(imgs, np.float32)

        predictions_val = predictions.eval(feed_dict={x: data})

        print(predictions_val)

def main():
    test_network(sys.argv[1])

main()
