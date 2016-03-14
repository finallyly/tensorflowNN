#coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 50
NUM_EPOCHS = 10000

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y_, 1), tf.arg_max(y, 1)), tf.float32))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for i in range(NUM_EPOCHS):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})