import tensorflow as tf
import numpy as np
import sys


input_x = np.array([[float(sys.argv[1]), float(sys.argv[2])]])

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model/or_model.ckpt.meta')
    saver.restore(sess, "model/or_model.ckpt")

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    output = graph.get_tensor_by_name("output:0")

    result = sess.run(output, feed_dict={x: input_x})
    if result[0][0] <= 0.5:
        print("0")
    else:
        print("1")
