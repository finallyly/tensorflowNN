#coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1.0 / np.sqrt(shape[0]), seed=0)
    return tf.Variable(initial)

def bias_variable(shape):
    return tf.Variable(np.zeros(shape, dtype=np.float32))

class FCNN(object):
    """
    fully connected neural network
    with one hidden layer
    """
    def __init__(self, input_dim, num_classes, size_hidden_layer):
        self.x = tf.placeholder(tf.float32, [None, input_dim])
        self.y_ = tf.placeholder(tf.float32, [None, num_classes])

        W_h1 = weight_variable([input_dim, size_hidden_layer])
        b_h1 = bias_variable([size_hidden_layer])
        h1 = tf.nn.relu(tf.matmul(self.x, W_h1) + b_h1)

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        h1_dropout = tf.nn.dropout(h1, self.keep_prob)

        W_softmax = weight_variable([size_hidden_layer, num_classes])
        b_softmax = bias_variable([num_classes])
        self.y = tf.nn.softmax(tf.matmul(h1_dropout, W_softmax) + b_softmax)

        cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_predictions = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    def fit(self, x, y, batch_size=50, num_epochs=1000, keep_prob=0.5):
        with tf.Session() as sess:
            with sess.as_default():
                tf.initialize_all_variables().run()
                for i in range(num_epochs):
                    batch_x = x[i*batch_size: (i+1)*batch_size]
                    batch_y = y[i*batch_size: (i+1)*batch_size]
                    assert batch_x.shape[0] == batch_size
                    if i%100 == 0:
                        accuracy = self.accuracy.eval(feed_dict={self.x: batch_x, self.y_: batch_y, self.keep_prob: 1.0})
                        print 'step {s}, accuracy on the training batch is {a:.2f}%'.format(s=i, a=accuracy*100.0)
                    self.train_step.run(feed_dict={self.x: batch_x, self.y_: batch_y, self.keep_prob: 1.0})
    def transform(self, x):
        with tf.Session() as sess:
            with sess.as_default():
                tf.initialize_all_variables().run()
                y = self.y.eval(feed_dict={self.x: x, self.keep_prob: 1.0})
                labels = np.argmax(y, axis=1)
                return labels

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    fcnn = FCNN(input_dim=784, num_classes=10, size_hidden_layer=1000)
    fcnn.fit(mnist.train.images, mnist.train.labels, batch_size=50, num_epochs=10000, keep_prob=0.5)
    prediction = fcnn.transform(mnist.test.images)
    accuracy = np.mean(prediction == np.argmax(mnist.test.labels, 1))
    print 'test accuracy is {a:.2f}%'.format(a=accuracy*100.0)