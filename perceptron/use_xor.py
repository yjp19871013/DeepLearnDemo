import tensorflow as tf
import numpy as np
import sys


input_x = np.array([[float(sys.argv[1]), float(sys.argv[2])]])

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model/nand_model.ckpt.meta')
    saver.restore(sess, "model/nand_model.ckpt")

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    output = graph.get_tensor_by_name("output:0")

    nand_result = sess.run(output, feed_dict={x: input_x})
    if nand_result[0][0] <= 0.5:
        nand_result = 0
    else:
        nand_result = 1

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model/or_model.ckpt.meta')
    saver.restore(sess, "model/or_model.ckpt")

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    output = graph.get_tensor_by_name("output:0")

    or_result = sess.run(output, feed_dict={x: input_x})
    if or_result[0][0] <= 0.5:
        or_result = 0
    else:
        or_result = 1

and_input = np.array([[nand_result, or_result]])
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model/and_model.ckpt.meta')
    saver.restore(sess, "model/and_model.ckpt")

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    output = graph.get_tensor_by_name("output:0")

    result = sess.run(output, feed_dict={x: and_input})

    if result[0][0] <= 0.5:
        print("0")
    else:
        print("1")
