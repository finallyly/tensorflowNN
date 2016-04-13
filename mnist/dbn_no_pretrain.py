# coding=utf-8
import tensorflow as tf
import numpy as np
import cPickle
import copy
import Image
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from util import tile_raster_images
from tf_util import Struct, GraphWrapper, iterate_dataset, ordinal_to_onehot
from rbm import RBM


class DBN(object):
    def __init__(self, rbm_layers, n_visible, num_classes, learning_rate=1e-3, batch_size=50, num_epochs=1000, probe_epochs=50):
        """
        :param rbm_layers: list of consecutive rbm layers, properly pretrained
        :return:
        """
        self.rbm_layers = rbm_layers
        self.n_visible = n_visible
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.probe_epochs = probe_epochs
        self.params = {}
    
    def __build_graph__(self):
        print 'building graph...'
        n_rbm_layers = len(self.rbm_layers)
        graph = tf.Graph()
        with graph.as_default():
            x = tf.placeholder(tf.float32, [None, self.n_visible])
            y_ = tf.placeholder(tf.float32, [None, self.num_classes])
            v = x
            W_list = []
            bias_list = []
            # rbm layers
            for i, rbm in zip(range(n_rbm_layers), self.rbm_layers):
                W = tf.Variable(initial_value=rbm.params['W'], trainable=True)
                c = tf.Variable(initial_value=rbm.params['c'], trainable=True)
                print '{i}th rbm layer, n_visible {v}, n_hidden {h}'\
                    .format(i=i, v=rbm.params['W'].shape[0], h=rbm.params['W'].shape[1])
                h = tf.sigmoid(tf.matmul(v, W) + c)
                W_list.append(W)
                bias_list.append(c)
                v = h
            # softmax layer
            n_hidden_last_rbm = self.rbm_layers[-1].params['c'].shape[0]
            print 'n_hidden_last_rbm = {n}'.format(n=n_hidden_last_rbm)
            initial_W = np.float32(np.random.uniform(
                low=-4 * np.sqrt(6.0 / (n_hidden_last_rbm + self.num_classes)),
                high=4 * np.sqrt(6.0 / (n_hidden_last_rbm + self.num_classes)),
                size=(n_hidden_last_rbm, self.num_classes)
            ))
            W = tf.Variable(initial_W)
            b = tf.Variable(tf.zeros([self.num_classes]))
            y = tf.nn.softmax(tf.matmul(v, W) + b)
            W_list.append(W)
            bias_list.append(b)
            cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.arg_max(y, 1)), tf.float32))
            train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy)
            init_vars = tf.initialize_all_variables()
        phr = Struct(x=x, y_=y_)
        var = Struct(W_list=W_list, bias_list=bias_list)
        tsr = Struct(y=y, cross_entropy=cross_entropy, accuracy=accuracy)
        ops = Struct(train_step=train_step, init_vars=init_vars)
        return GraphWrapper(graph, phr, var, tsr, ops)
    
    def finetune(self, x, y):
        num_samples = x.shape[0]
        n_visible = x.shape[1]
        num_classes = y.max() + 1
        print '{s} samples in R^{v}, {c} classes'.format(s=num_samples, v=n_visible, c=num_classes)
        y = ordinal_to_onehot(y)
        G = self.__build_graph__()
        sess = tf.Session(graph=G.graph)
        with sess.as_default():
            np.random.seed(3)
            G.ops.init_vars.run()
            dataset = iterate_dataset(x, y, self.batch_size)
            for i in range(self.num_epochs):
                batch_x, batch_y = dataset.next()
                if i % self.probe_epochs == 0:
                    accuracy = G.tsr.accuracy.eval(feed_dict={G.phr.x: batch_x, G.phr.y_: batch_y})
                    print 'step {i}, accuracy on the batch {a:.2f}%'.format(i=i, a=accuracy*100.0)
                G.ops.train_step.run(feed_dict={G.phr.x: batch_x, G.phr.y_: batch_y})
            for k, v in G.var.iteritems():
                self.params[k] = sess.run(v)
    
    def fit(self, x, targets):
        self.finetune(x, targets)

    def predict_proba(self, x):
        num_samples = x.shape[0]
        n_visible = x.shape[1]
        print '{s} samples in R^{v}'.format(s=num_samples, v=n_visible)
        G = self.__build_graph__()
        sess = tf.Session(graph=G.graph)
        with sess.as_default():
            feed_dict = dict([(k, v) for k, v in zip(G.var.W_list, self.params['W_list'])])
            feed_dict.update(dict([(k, v) for k, v in zip(G.var.bias_list, self.params['bias_list'])]))
            feed_dict.update({G.phr.x: x})
            predict = G.tsr.y.eval(feed_dict=feed_dict)
            return predict

    def predict(self, x):
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)


def pretrain_rbm_layers(v, validation_v=None, n_hidden=[], gibbs_steps=[], batch_size=[], num_epochs=[], learning_rate=[], probe_epochs=[]):
    """
    Fake pre-training, just randomly initialising the weights of RBM layers
    :param v:
    :param validation_v:
    :param n_hidden:
    :param gibbs_steps:
    :param batch_size:
    :param num_epochs:
    :param learning_rate:
    :param probe_epochs:
    :return:
    """
    rbm_layers = []
    n_rbm = len(n_hidden)
    # create rbm layers
    for i in range(n_rbm):
        rbm = RBM(n_hidden=n_hidden[i],
                    gibbs_steps=gibbs_steps[i],
                    batch_size=batch_size[i],
                    num_epochs=num_epochs[i],
                    learning_rate=learning_rate[i],
                    probe_epochs=probe_epochs[i])
        rbm_layers.append(rbm)
    # pretrain rbm layers
    n_v = v.shape[1]
    for rbm, i in zip(rbm_layers, range(len(rbm_layers))):
        print '### pretraining RBM Layer {i}'.format(i=i)
        n_h = n_hidden[i]
        initial_W = np.float32(np.random.uniform(
            low=-4 * np.sqrt(6.0 / (n_h + n_v)),
            high=4 * np.sqrt(6.0 / (n_h + n_v)),
            size=(n_v, n_h)
        ))
        rbm.params['W'] = initial_W
        rbm.params['c'] = np.zeros((n_h, ), np.float32)
        n_v = n_h
    return rbm_layers
    
def test_dbn():
    pass
    
if __name__ == "__main__":
    plt.close('all')
    np.random.seed(1)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    train_x = np.float32(mnist.train.images > 0)
    train_y = mnist.train.labels
    validation_x = np.float32(mnist.validation.images[np.random.permutation(mnist.validation.images.shape[0])][0:1000] > 0)

    n_hidden = [500, 500]
    learning_rate = [1e-3] * 2
    gibbs_steps = [10] * 2
    batch_size = [50] * 2
    num_epochs = [2000] * 2
    probe_epochs = [100] * 2
    rbm_layers = pretrain_rbm_layers(train_x,
                                     validation_x,
                                     n_hidden=n_hidden,
                                     gibbs_steps=gibbs_steps,
                                     batch_size=batch_size,
                                     num_epochs=num_epochs,
                                     learning_rate=learning_rate,
                                     probe_epochs=probe_epochs)

    dbn = DBN(rbm_layers=rbm_layers,
              n_visible=28*28,
              num_classes=10,
              learning_rate=1e-3,
              batch_size=50,
              num_epochs=10000,
              probe_epochs=100)
    dbn.fit(train_x, train_y)
    # test
    test_x = np.float32(mnist.test.images > 0)
    test_y = mnist.test.labels
    predicted_labels = dbn.predict(test_x)
    accuracy = np.mean(predicted_labels == test_y)
    print '{s} test samples, accuracy {a:.2f}%'.format(s=test_x.shape[0], a=accuracy*100.0)