import tensorflow as tf
import matplotlib.pyplot as plt


def train_logical_gate(train_x, train_labels, save_path, learning_rate=0.9, batch=100):
    # 输入参数个数
    input_num = 2

    # 输出个数
    output_num = 1

    w = tf.Variable(tf.truncated_normal([input_num, output_num], stddev=0.01))
    b = tf.Variable(tf.truncated_normal([output_num], 0.1, stddev=0.01))
    x = tf.placeholder(tf.float32, shape=[None, input_num], name="x")
    y = tf.placeholder(tf.float32, shape=[None, output_num])

    output = tf.sigmoid(tf.matmul(x, w) + b, name="output")
    loss = tf.reduce_mean(tf.square(output - y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=1)
        sess.run(tf.global_variables_initializer())

        for i in range(batch):
            sess.run([train_step], feed_dict={x: train_x, y: train_labels})

        saver.save(sess, save_path)

        w_learned = sess.run(w)
        b_learned = sess.run(b)
        print("w: " + str(w_learned))
        print("b: " + str(b_learned))

        plt.scatter(train_x[:, 0], train_x[:, 1])
        plt.plot(train_x[:, 0], (-b_learned - train_x[:, 0] * w_learned[0][0]) / w_learned[1][0])
        plt.show()
