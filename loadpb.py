import tensorflow as tf
import numpy as np
import sys
import os

def recognize(img_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open("inception_v1.pb", "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            input = sess.graph.get_tensor_by_name("input:0")
            output = sess.graph.get_tensor_by_name("output:0")

            imgfloat = tf.cast(tf.image.decode_jpeg(tf.read_file(img_path), channels=3), dtype=tf.float32)
            img = tf.subtract(tf.multiply(tf.div(tf.image.resize_images(tf.expand_dims(imgfloat, 0), (224, 224), method=0), 255.0), 2), 1.0)

            img_out_softmax = sess.run(output, feed_dict={input:img.eval()})

            file = open("labels.txt")
            labels = file.readlines()
            prediction_labels = np.argmax(img_out_softmax, axis=1)
            print(labels[prediction_labels[0]])

recognize(sys.argv[1])
