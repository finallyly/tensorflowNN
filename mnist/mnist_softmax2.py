#coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class GraphWrapper(object):
    def __init__(self, graph, phr, var, tsr, ops):
        self.graph = graph
        self.phr = phr
        self.var = var
        self.tsr = tsr
        self.ops = ops


class SoftmaxRegressor(object):
    def __init__(self, learning_rate, batch_size, num_epochs):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.params = {}

    def __build_graph__(self, input_dim=None, num_classes=None, var_val=None):
        graph = tf.Graph()
        with graph.as_default():
            if var_val is not None:
                input_dim = var_val['W'].shape[0]
                num_classes = var_val['W'].shape[1]
                W = tf.Variable(var_val['W'])
                b = tf.Variable(var_val['b'])
            else:
                W = tf.Variable(tf.zeros([input_dim, num_classes]))
                b = tf.Variable(tf.zeros([num_classes]))
            x = tf.placeholder(tf.float32, [None, input_dim])
            y_ = tf.placeholder(tf.float32, [None, num_classes])
            y = tf.nn.softmax(tf.matmul(x, W) + b)
            cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.arg_max(y, 1)), tf.float32))
            train_step = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(cross_entropy)
            init_vars = tf.initialize_all_variables()
        phr = {'x': x, 'y_': y_}
        var = {'W': W, 'b': b}
        tsr = {'y': y, 'cross_entropy': cross_entropy, 'accuracy': accuracy}
        ops = {'train_step': train_step, 'init_vars': init_vars}
        return GraphWrapper(graph, phr, var, tsr, ops)

    def fit(self, x, y, verbose=True):
        num_samples = x.shape[0]
        input_dim = x.shape[1]
        num_classes = y.shape[1]
        G = self.__build_graph__(input_dim, num_classes)
        sess = tf.Session(graph=G.graph)
        with sess.as_default():
            G.ops['init_vars'].run()
            head = 0
            indices = range(num_samples)
            for i in range(self.num_epochs):
                if head + self.batch_size > num_samples:
                    indices = np.random.permutation(num_samples)
                    head = 0
                selected = indices[head: head + self.batch_size]
                head += self.batch_size
                batch_x = x[selected]
                batch_y = y[selected]
                if verbose:
                    if i % 100 == 0:
                        accuracy = G.tsr['accuracy'].eval(feed_dict={G.phr['x']: batch_x, G.phr['y_']: batch_y})
                        print 'step {s}, accuracy on the training batch is {a:.2f}%'.format(s=i, a=accuracy * 100.0)
                G.ops['train_step'].run(feed_dict={G.phr['x']: batch_x, G.phr['y_']: batch_y})
            for k, v in G.var.iteritems():
                self.params[k] = G.var[k].eval()              
                
    def predict_proba(self, x):
        G = self.__build_graph__(var_val=self.params)
        sess = tf.Session(graph=G.graph)
        with sess.as_default():
            G.ops['init_vars'].run()
            return G.tsr['y'].eval(feed_dict={G.phr['x']: x})

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    softmax_regressor = SoftmaxRegressor(
        learning_rate=1e-4,
        batch_size=50,
        num_epochs=1000)
    softmax_regressor.fit(mnist.train.images, mnist.train.labels)
    prediction = softmax_regressor.predict_proba(mnist.test.images)
    accuracy = np.mean(np.argmax(prediction, 1) == np.argmax(mnist.test.labels, 1))
    print 'test accuracy is {a:.2f}%'.format(a=accuracy * 100.0)

