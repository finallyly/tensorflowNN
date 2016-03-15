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
    def __init__(self, input_dim=None, num_classes=None, size_hidden_layer=None, learning_rate=0.0001, batch_size=50, num_epochs=1000, keep_probability=0.5, random_seed=None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.size_hidden_layer = size_hidden_layer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.keep_probability = keep_probability
        self.random_seed = random_seed

    def set_params(self, input_dim=None, num_classes=None, size_hidden_layer=None, learning_rate=None, batch_size=None, num_epochs=None, keep_probability=None, random_seed=None):
        if input_dim is not None:
            self.input_dim = input_dim
        if num_classes is not None:
            self.num_classes = num_classes
        if size_hidden_layer is not None:
            self.size_hidden_layer = size_hidden_layer
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if batch_size is not None:
            self.batch_size = batch_size
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if keep_probability is not None:
            self.keep_probability = keep_probability
        if random_seed is not None:
            self.random_seed = random_seed

    def print_params(self):
        print '-- parameters --'
        print 'input_dim: {v}'.format(v=self.input_dim)
        print 'num_classes: {v}'.format(v=self.num_classes)
        print 'size_hidden_layer: {v}'.format(v=self.size_hidden_layer)
        print 'learning_rate: {v}'.format(v=self.learning_rate)
        print 'batch_size: {v}'.format(v=self.batch_size)
        print 'num_epochs: {v}'.format(v=self.num_epochs)
        print 'keep_probability: {v}'.format(v=self.keep_probability)
        print 'random_seed: {v}'.format(v=self.random_seed)

    def __build_graph__(self):
        self.x = tf.placeholder(tf.float32, [None, self.input_dim])
        self.y_ = tf.placeholder(tf.float32, [None, self.num_classes])

        W_h1 = weight_variable([self.input_dim, self.size_hidden_layer])
        b_h1 = bias_variable([self.size_hidden_layer])
        h1 = tf.nn.relu(tf.matmul(self.x, W_h1) + b_h1)

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        h1_dropout = tf.nn.dropout(h1, self.keep_prob, seed=1)

        W_softmax = weight_variable([self.size_hidden_layer, self.num_classes])
        b_softmax = bias_variable([self.num_classes])
        self.y = tf.nn.softmax(tf.matmul(h1_dropout, W_softmax) + b_softmax)

        self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
        correct_predictions = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def fit(self, x, y, verbose=True):
        self.__build_graph__()
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        if verbose:
            self.print_params()
        self.sess = tf.Session()
        num_sample = x.shape[0]
        with self.sess.as_default():
            tf.initialize_all_variables().run()
            head = 0
            indices = range(num_sample)
            for i in range(self.num_epochs):
                if head + self.batch_size > num_sample:
                    indices = np.random.permutation(num_sample)
                    head = 0
                selected = indices[head : head+self.batch_size]
                head += self.batch_size
                batch_x = x[selected]
                batch_y = y[selected]
                assert batch_x.shape[0] == self.batch_size
                if verbose:
                    if i%100 == 0:
                        accuracy = self.accuracy.eval(feed_dict={self.x: batch_x, self.y_: self.__ordinal_to_onehot__(batch_y), self.keep_prob: 1.0})
                        print 'step {s}, accuracy on the training batch is {a:.2f}%'.format(s=i, a=accuracy*100.0)
                self.train_step.run(feed_dict={self.x: batch_x, self.y_: self.__ordinal_to_onehot__(batch_y), self.keep_prob: self.keep_probability})

    def transform(self, x):
        with self.sess.as_default():
            y = self.y.eval(feed_dict={self.x: x, self.keep_prob: 1.0})
            labels = np.argmax(y, axis=1)
            return labels

    def __ordinal_to_onehot__(self, y):
        yy = np.zeros((y.shape[0], self.num_classes), np.float32)
        yy[np.arange(y.shape[0]), y] = 1.0
        return yy

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    # fcnn = FCNN(input_dim=784, num_classes=10, size_hidden_layer=1000, learning_rate=1e-4, batch_size=50, num_epochs=1000, keep_probability=0.5)
    fcnn = FCNN()
    est_params = {'input_dim': 784,
                  'num_classes': 10,
                  'size_hidden_layer': 1000,
                  'learning_rate': 1e-4,
                  'batch_size': 50,
                  'num_epochs': 1000,
                  'keep_probability': 0.5,
                  'random_seed': 0}
    fcnn.set_params(**est_params)
    fcnn.fit(mnist.train.images, mnist.train.labels)
    prediction = fcnn.transform(mnist.test.images)
    accuracy = np.mean(prediction == mnist.test.labels)
    print 'test accuracy is {a:.2f}%'.format(a=accuracy*100.0)