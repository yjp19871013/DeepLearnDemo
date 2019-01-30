import mnist
import numpy as np
import tensorflow as tf

learning_rate = 0.05
batch = 10000
batch_size = 1000
save_path = "model/mnist.ckpt"

(train_img, train_label), (test_img, test_label) = mnist.load_mnist(normalize=True, flatten=True, one_hot_label=True)

input_num = train_img.shape[1]
output_num = 10

input_img = tf.placeholder(tf.float32, [None, input_num], name="input")
output_actual = tf.placeholder(tf.float32, [None, output_num], name="input")

# 隐藏层
w = tf.Variable(tf.truncated_normal([input_num, output_num], stddev=0.01))
b = tf.Variable(tf.truncated_normal([output_num], stddev=0.01))
layer = tf.sigmoid(tf.matmul(input_img, w) + b)

# 输出层
output = tf.nn.softmax(layer, name="output")

cross_entropy = tf.reduce_mean(-tf.reduce_sum(output_actual * tf.log(output)))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=1)
    sess.run(tf.global_variables_initializer())

    for i in range(batch):
        batch_train_mask = np.random.choice(train_img.shape[0], batch_size)
        sess.run([train_step], feed_dict={input_img: train_img[batch_train_mask],
                                          output_actual: train_label[batch_train_mask]})

    saver.save(sess, save_path)
