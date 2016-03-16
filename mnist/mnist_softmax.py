#coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class SoftmaxRegressor(object):
    def __init__(self, input_dim, num_classes, learning_rate, batch_size, num_epochs):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def __build_graph__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, self.input_dim])
            self.y_ = tf.placeholder(tf.float32, [None, self.num_classes])
            W = tf.Variable(tf.zeros([self.input_dim, self.num_classes]))
            b = tf.Variable(tf.zeros([self.num_classes]))
            self.y = tf.nn.softmax(tf.matmul(self.x, W) + b)
            cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.y_, 1), tf.arg_max(self.y, 1)), tf.float32))
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
            self.init_vars = tf.initialize_all_variables()

    def fit(self, x, y, verbose=True):
        self.__build_graph__()
        num_sample = x.shape[0]
        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            self.init_vars.run()
            head = 0
            indices = range(num_sample)
            for i in range(self.num_epochs):
                if head + self.batch_size > num_sample:
                    indices = np.random.permutation(num_sample)
                    head = 0
                selected = indices[head: head + self.batch_size]
                head += self.batch_size
                batch_x = x[selected]
                batch_y = y[selected]
                if verbose:
                    if i % 100 == 0:
                        accuracy = self.accuracy.eval(feed_dict={self.x: batch_x, self.y_: batch_y})
                        print 'step {s}, accuracy on the training batch is {a:.2f}%'.format(s=i, a=accuracy * 100.0)
                self.train_step.run(feed_dict={self.x: batch_x, self.y_: batch_y})

    def predict_proba(self, x):
        with self.sess.as_default():
            return self.y.eval(feed_dict={self.x: x})


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    softmax_regressor = SoftmaxRegressor(
        input_dim=784,
        num_classes=10,
        learning_rate=1e-4,
        batch_size=50,
        num_epochs=10000)
    softmax_regressor.fit(mnist.train.images, mnist.train.labels)
    prediction = softmax_regressor.predict_proba(mnist.test.images)
    accuracy = np.mean(np.argmax(prediction, 1) == np.argmax(mnist.test.labels, 1))
    print 'test accuracy is {a:.2f}%'.format(a=accuracy * 100.0)