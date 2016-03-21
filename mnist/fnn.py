#coding=utf-8
import numpy as np
import tensorflow as tf
import cPickle
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1.0 / np.sqrt(shape[0]), seed=0)
    return tf.Variable(initial)

def bias_variable(shape):
    return tf.Variable(np.zeros(shape, dtype=np.float32))

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

class FCNN(object):
    def __init__(self):
        # 模型的参数,对应于Graph中的trainable variables
        self.params = {}
        # Graph and session for online running
        self.updated = False
        self.G_run = None
        self.sess_run = None

    def __build_graph__(self, input_dim=None, num_classes=None, size_hidden_layer=None, var_val=None):
        """
        Define the structure of the graph
        :param input_dim:
        :param num_classes:
        :param size_hidden_layer:
        :param var_val:
        :return:
        """
        graph = tf.Graph()
        with graph.as_default():
            if var_val is not None:
                input_dim = var_val['W_h1'].shape[0]
                num_classes = var_val['W_smx'].shape[1]
                # 1st hidden layer
                W_h1 = tf.Variable(var_val['W_h1'])
                b_h1 = tf.Variable(var_val['b_h1'])
                # softmax layer
                W_smx = tf.Variable(var_val['W_smx'])
                b_smx = tf.Variable(var_val['b_smx'])
            else:
                # 1st hidden layer
                W_h1 = weight_variable([input_dim, size_hidden_layer])
                b_h1 = bias_variable([size_hidden_layer])
                # softmax layer
                W_smx = weight_variable([size_hidden_layer, num_classes])
                b_smx = bias_variable([num_classes])
            x = tf.placeholder(tf.float32, [None, input_dim])
            y_ = tf.placeholder(tf.float32, [None, num_classes])
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            learning_rate = tf.placeholder(tf.float32)
            # the first hidden layer
            h1 = tf.nn.relu(tf.matmul(x, W_h1) + b_h1)
            # dropout layer
            h1_dropout = tf.nn.dropout(h1, keep_prob, seed=1)
            # softmax layer
            y = tf.nn.softmax(tf.matmul(h1_dropout, W_smx) + b_smx)
            # evaluation
            cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
            correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            init_vars = tf.initialize_all_variables()
        phr = Struct(x=x, y_=y_, keep_prob=keep_prob, learning_rate=learning_rate)
        var = Struct(W_h1 =W_h1, b_h1=b_h1, W_smx=W_smx, b_smx=b_smx)
        tsr = Struct(y=y, accuracy=accuracy)
        ops = Struct(train_step=train_step, init_vars=init_vars)
        return GraphWrapper(graph, phr, var, tsr, ops)

    def fit(self, x, y, size_hidden_layer, batch_size=50, num_epochs=1e4, learning_rate=1e-3, keep_prob=0.5, verbose=True):
        """
        :param x: np.ndarray, each row stores an instance
        :param y: one-dimensional np.ndarray, 0 .. num_classes-1
        """
        num_samples = x.shape[0]
        input_dim = x.shape[1]
        y = np.int32(y)
        num_classes = y.max() + 1
        y = self.__ordinal_to_onehot__(y)
        G = self.__build_graph__(input_dim, num_classes, size_hidden_layer)
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
                        accuracy = G.tsr.accuracy.eval(feed_dict={G.phr.x: batch_x, G.phr.y_: batch_y, G.phr.keep_prob: 1.0})
                        print 'step {s}, accuracy on the training batch is {a:.2f}%'.format(s=i, a=accuracy * 100.0)
                G.ops.train_step.run(feed_dict={G.phr.x: batch_x, G.phr.y_: batch_y,
                                                G.phr.learning_rate: learning_rate,
                                                G.phr.keep_prob: keep_prob})
            for k, v in G.var.iteritems():
                self.params[k] = G.var[k].eval()
        self.updated = True

    def predict_proba(self, x):
        """
        :param x:
        :return: np.2darray, each row gives the proba of an instance belonging to each class
        """
        if len(self.params) < 1:
            raise Exception("empty model")
        if self.updated:
            self.G_run = self.__build_graph__(var_val=self.params)
            if self.sess_run is not None: self.sess_run.close()
            self.sess_run = tf.Session(graph=self.G_run.graph)
            self.sess_run.run(self.G_run.ops.init_vars)
            self.updated = False
        proba = self.sess_run.run(self.G_run.tsr.y, feed_dict={self.G_run.phr.x: x, self.G_run.phr.keep_prob: 1.0})
        return proba

    def predict(self, x):
        """
        :param x:
        :return: one-dimensional np.ndarray, ordinal label
        """
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
    fcnn = FCNN()
    fcnn.fit(
        x=mnist.train.images,
        y=mnist.train.labels,
        size_hidden_layer=1000,
        learning_rate=1e-3,
        keep_prob=0.5,
        batch_size=50,
        num_epochs=1000
    )
    accuracy = np.mean(fcnn.predict(mnist.test.images) == mnist.test.labels)
    print 'test accuracy is {a:.2f}%'.format(a=accuracy * 100.0)
    fcnn.save("model")