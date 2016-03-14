#coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1.0 / np.sqrt(shape[0]), seed=0)
    return tf.Variable(initial)

def bias_variable(shape):
    return tf.Variable(np.zeros(shape, dtype=np.float32))

DIM_INPUT = 784
NUM_CLASSES = 10
SIZE_HIDDEN_LAYER1 = 1000
BATCH_SIZE = 50
NUM_EPOCHS = 10000
KEEP_PROB = 0.5

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, DIM_INPUT])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    W_h1 = weight_variable([DIM_INPUT, SIZE_HIDDEN_LAYER1])
    b_h1 = bias_variable([SIZE_HIDDEN_LAYER1])
    h1 = tf.nn.relu(tf.matmul(x, W_h1) + b_h1)

    keep_prob = tf.placeholder(tf.float32)
    h1_dropout = tf.nn.dropout(h1, keep_prob)

    W_softmax = weight_variable([SIZE_HIDDEN_LAYER1, NUM_CLASSES])
    b_softmax = bias_variable([NUM_CLASSES])
    y = tf.nn.softmax(tf.matmul(h1_dropout, W_softmax) + b_softmax)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    sess = tf.Session()
    with sess.as_default():
        tf.initialize_all_variables().run()
        for i in range(NUM_EPOCHS):
            batch = mnist.train.next_batch(BATCH_SIZE)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: KEEP_PROB})
            if i%(NUM_EPOCHS/50) == 0:
                validation_accuracy = accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
                print 'step {i}, validation accuracy is {a:.2f}%'.format(i=i, a=100.0*validation_accuracy)
        test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print 'test accuracy is {a:.2f}%'.format(a=100.0*test_accuracy)
    sess.close()