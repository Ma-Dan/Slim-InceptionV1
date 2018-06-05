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
import os
import glob


slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets import inception_v1

def test_network():
    x = tf.placeholder("float", shape=[None, 224, 224, 3], name='input')
    with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
        logits, end_points = inception_v1.inception_v1(x, num_classes=2, dropout_keep_prob=1.0, is_training=False)
    predictions = end_points["Predictions"]

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "train.ckpt")

        path = './picture/'
        w = 224
        h = 224
        c = 3
        cate   = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
        imgs   = []
        labels = []
        for idx, folder in enumerate(cate):
            for im in glob.glob(folder + '/*.jpg'):
                print('reading the image: %s' % (im))
                img = io.imread(im)
                img = transform.resize(img, (w, h, c))
                imgs.append(img)
                labels.append([1, 0] if idx == 0 else [0, 1])
                break
            break

        data = np.asarray(imgs, np.float32)
        label = np.asarray(labels, np.int32)

        predictions_val = predictions.eval(feed_dict={x: data})

        print(predictions_val)

def main():
    test_network()

main()
