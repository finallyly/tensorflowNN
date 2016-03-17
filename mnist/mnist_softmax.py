#coding=utf-8
import numpy as np
import tensorflow as tf
import cPickle
from tensorflow.examples.tutorials.mnist import input_data

class Struct(dict):
    def __init__(self, **kwargs):
        super(Struct, self).__init__(**kwargs)
        self.__dict__ = self


class GraphWrapper(object):
    def __init__(self, graph, phr, var, tsr, ops):
        self.graph = graph
        self.phr = phr
        self.var = var
        self.tsr = tsr
        self.ops = ops


class SoftmaxRegressor(object):
    def __init__(self):
        self.params = {}
        # Graph and session for online running
        self.updated = False
        self.G_run = None
        self.sess_run = None
        
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
            learning_rate = tf.placeholder(tf.float32)
            y = tf.nn.softmax(tf.matmul(x, W) + b)
            cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.arg_max(y, 1)), tf.float32))
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
            init_vars = tf.initialize_all_variables()
        phr = Struct(x=x, y_=y_, learning_rate=learning_rate)
        var = Struct(W=W, b=b)
        tsr = Struct(y=y, cross_entropy=cross_entropy, accuracy=accuracy)
        ops = Struct(train_step=train_step, init_vars=init_vars)
        return GraphWrapper(graph, phr, var, tsr, ops)

    def fit(self, x, y, learning_rate, batch_size, num_epochs, verbose=True):
        """

        :param x: np.ndarray, each row stores an instance
        :param y: one-dimensional np.ndarray, 0 .. num_classes-1
        :param learning_rate:
        :param batch_size:
        :param num_epochs:
        :param verbose:
        :return:
        """
        num_samples = x.shape[0]
        input_dim = x.shape[1]
        num_classes = y.max() + 1
        y = self.__ordinal_to_onehot__(y)
        G = self.__build_graph__(input_dim, num_classes)
        sess = tf.Session(graph=G.graph)
        with sess.as_default():
            G.ops.init_vars.run()
            head = 0
            indices = range(num_samples)
            for i in range(num_epochs):
                if head + batch_size > num_samples:
                    indices = np.random.permutation(num_samples)
                    head = 0
                selected = indices[head: head + batch_size]
                head += batch_size
                batch_x = x[selected]
                batch_y = y[selected]
                if verbose:
                    if i % 100 == 0:
                        accuracy = G.tsr.accuracy.eval(feed_dict={G.phr.x: batch_x, G.phr.y_: batch_y})
                        print 'step {s}, accuracy on the training batch is {a:.2f}%'.format(s=i, a=accuracy * 100.0)
                G.ops.train_step.run(feed_dict={G.phr.x: batch_x, G.phr.y_: batch_y, G.phr.learning_rate: learning_rate})
            for k, v in G.var.iteritems():
                self.params[k] = G.var[k].eval()
        self.updated = True         
                
    def predict_proba(self, x):
        if len(self.params) < 1:
            raise Exception("empty model")
        if self.updated:
            self.G_run = self.__build_graph__(var_val=self.params)
            if self.sess_run is not None: self.sess_run.close()
            self.sess_run = tf.Session(graph=self.G_run.graph)
            self.sess_run.run(self.G_run.ops.init_vars)
            self.updated = False
        proba = self.sess_run.run(self.G_run.tsr.y, feed_dict={self.G_run.phr.x: x})
        return proba

    def predict(self, x):
        proba = self.predict_proba(x)
        return np.argmax(proba, 1)

    def save(self, path):
        if len(self.params) < 1:
            raise Exception("empty model")
        with open(path, "wb") as f:
            cPickle.dump(self.params, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.params = cPickle.load(f)
        self.updated = True

    def __ordinal_to_onehot__(self, y):
        yy = np.zeros((y.shape[0], y.max()+1), dtype=np.int32)
        yy[np.arange(y.shape[0]), y] = 1
        return yy
    
    def __del__(self):
        if self.sess_run is not None:
            self.sess_run.close()

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    softmax_regressor = SoftmaxRegressor()
    softmax_regressor.fit(mnist.train.images,
                      mnist.train.labels,
                      learning_rate=1e-3,
                      batch_size=50,
                      num_epochs=10000)
    accuracy = np.mean(softmax_regressor.predict(mnist.test.images) == mnist.test.labels)
    print 'test accuracy is {a:.2f}%'.format(a=accuracy * 100.0)

    softmax_regressor.save("model")

    softmax_regressor.load("model")
    
    accuracy = np.mean(softmax_regressor.predict(mnist.test.images) == mnist.test.labels)
    print 'test accuracy is {a:.2f}%'.format(a=accuracy * 100.0)
    
    

