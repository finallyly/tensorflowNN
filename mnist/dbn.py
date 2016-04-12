# coding=utf-8
import tensorflow as tf
import numpy as np
import cPickle
import copy
import Image
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from util import tile_raster_images
from tf_util import Struct, GraphWrapper, sigmoid
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
        self.params = []
    
    def __build_graph__(self):
        n_rbm_layers = len(self.rbm_layers)
        graph = tf.Graph()
        with graph.as_default():
            x = tf.placeholder(tf.float32, [None, self.n_visible])
            y_ = tf.placeholder(tf.float32, [None, self.num_classes])
            v = x
            W_list = []
            bias_list = []
            # rbm layers
            for i, rbm in zip(range(n_rbm_layers, self.rbm_layers)):
                W = tf.Variable(initial_value=rbm.params['W'], trainable=True)
                c = tf.Variable(initial_value=rbm.params['c'], trainable=True)
                h = tf.sigmoid(tf.matmul(v, W) + c)
                W_list.append(W)
                bias_list.append(c)
                v = h
            # softmax layer
            initial_W = np.float32(np.random.uniform(
                low=-4 * np.sqrt(6.0 / (self.n_visible + self.num_classes)),
                high=4 * np.sqrt(6.0 / (self.n_visible + self.num_classes)),
                size=(self.n_visible, self.num_classes)
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
        pass
    
    def fit(self, v, targets):
        self.finetune(v, targets)


def pretrain_rbm_layers(v, validation_v=None, n_hidden=[], gibbs_steps=[], batch_size=[], num_epochs=[], learning_rate=[], probe_epochs=[]):
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
    input = v
    validation_input = validation_v
    for rbm, i in zip(rbm_layers, range(len(rbm_layers))):
        print '### pretraining RBM Layer {i}'.format(i=i)
        rbm.fit(input, validation_input)
        output = rbm.sample_h_given_v(input, rbm.params['W'], rbm.params['c'])
        if validation_input is not None:
            validation_output = rbm.sample_h_given_v(validation_input, rbm.params['W'], rbm.params['c'])
        else:
            validation_output = None
        input = output
        validation_input = validation_output
    return rbm_layers
    

if __name__ == "__main__":
    plt.close('all')
    np.random.seed(1)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    train_v = np.float32(mnist.train.images > 0)
    validation_v = np.float32(mnist.validation.images[np.random.permutation(mnist.validation.images.shape[0])][0:1000] > 0)

    n_hidden = [500, 300]
    learning_rate = [1e-2] * 2
    gibbs_steps = [10] * 2
    batch_size = [100] * 2
    num_epochs = [500] * 2
    probe_epochs = [50] * 2
    rbm_layers = pretrain_rbm_layers(train_v,
                                     validation_v,
                                     n_hidden=n_hidden,
                                     gibbs_steps=gibbs_steps,
                                     batch_size=batch_size,
                                     num_epochs=num_epochs,
                                     learning_rate=learning_rate,
                                     probe_epochs=probe_epochs)
