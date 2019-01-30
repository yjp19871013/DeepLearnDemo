import tensorflow as tf
import numpy as np
import mnist

(train_img, train_label), (test_img, test_label) = mnist.load_mnist(normalize=True, flatten=True, one_hot_label=True)


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model/mnist.ckpt.meta')
    saver.restore(sess, "model/mnist.ckpt")

    graph = tf.get_default_graph()
    input = graph.get_tensor_by_name("input:0")
    output = graph.get_tensor_by_name("output:0")

    result = sess.run(output, feed_dict={input: test_img})
    print(len(result[np.argmax(test_label, axis=1) != np.argmax(result, axis=1)]) / len(test_label))
