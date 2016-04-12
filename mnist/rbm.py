# coding=utf-8
import tensorflow as tf
import numpy as np
import cPickle
import copy
import Image
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from utils import tile_raster_images

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


class RBM(object):
    def __init__(self):
        # 模型的参数,对应于Graph中的trainable variables
        self.params = {}
        # Graph and session for online running
        self.updated = False
        self.G_run = None
        self.sess_run = None

    def __del__(self):
        if self.sess_run is not None:
            self.sess_run.close()

    def __build_graph__(self, n_visible=None, n_hidden=None, var_val=None):
        graph = tf.Graph()
        with graph.as_default():
            if var_val is not None:
                n_visible = var_val['W'].shape[0]
                n_hidden = var_val['W'].shape[1]
                W = tf.Variable(var_val['W'])
                b = tf.Variable(var_val['b'])
                c = tf.Variable(var_val['c'])
            else:
                initial_W = np.float32(np.random.uniform(
                    low=-4 * np.sqrt(6.0 / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6.0 / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ))
                W = tf.Variable(initial_value=initial_W, trainable=True)
                b = tf.Variable(np.zeros((n_visible,), np.float32), trainable=True)
                c = tf.Variable(np.zeros((n_hidden,), np.float32), trainable=True)
            v = tf.placeholder(tf.float32, [None, n_visible])
            v_sampling = tf.placeholder(tf.float32, [None, n_visible])
            learning_rate = tf.placeholder(tf.float32)
            loss = tf.reduce_mean(self.__calc_free_energy__(v, W, b, c)) - tf.reduce_mean(self.__calc_free_energy__(v_sampling, W, b, c))
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            init_vars = tf.initialize_all_variables()
        phr = Struct(v=v, v_sampling=v_sampling, learning_rate=learning_rate)
        var = Struct(W=W, b=b, c=c)
        tsr = Struct(loss=loss)
        ops = Struct(train_step=train_step, init_vars=init_vars)
        return GraphWrapper(graph, phr, var, tsr, ops)

    def fit(self, v, n_hidden, gibbs_steps=1, batch_size=50, num_epochs=10000, learning_rate=1e-3, validation_v=None, probe_epochs=50):
        """
        :param v: 2d np.ndarray, each row stores a sample
        :param n_hidden:
        :param gibbs_steps:
        :param batch_size:
        :param num_epochs:
        :param learning_rate:
        :param validation_v: 
        :param probe_epochs:
        :return:
        """
        if ((v == 0.0) | (v == 1.0)).sum() != v.shape[0] * v.shape[1]:
            raise Exception('v should be binary')
        n_visible = v.shape[1]
        msg = '{t} training samples'.format(t=v.shape[0])
        if validation_v is not None:
            msg += ', {v} validation samples'.format(v=validation_v.shape[0])
        print msg
        G = self.__build_graph__(n_visible=n_visible, n_hidden=n_hidden)
        sess = tf.Session(graph=G.graph)
        with sess.as_default():
            np.random.seed(3)
            G.ops.init_vars.run()
            dataset = self.iterate_dataset(v, batch_size)
            for i in range(num_epochs):
                batch_v = dataset.next()
                batch_v_sampling = self.gibbs_v(v0=batch_v, W=G.var.W.eval(), b=G.var.b.eval(), c=G.var.c.eval(), k=gibbs_steps)
                if i % probe_epochs == 0:
                    loss = G.tsr.loss.eval(feed_dict={G.phr.v: batch_v, G.phr.v_sampling: batch_v_sampling})
                    msg = 'step {i}, loss {l:.4f}'.format(i=i, l=loss)
                    if validation_v is not None:
                        # reconstruct the visible units by single step Gibbs sampling
                        reconstruct_v = self.gibbs_v(v0=validation_v, W=G.var.W.eval(), b=G.var.b.eval(), c=G.var.c.eval(), k=1)
                        mae = 1.0 * np.abs(reconstruct_v - validation_v).sum() / validation_v.shape[0]
                        msg += ", validation reconstruct MAE {e:.4f}".format(e=mae)
                    print msg
                G.ops.train_step.run(feed_dict={G.phr.v: batch_v, G.phr.v_sampling: batch_v_sampling, G.phr.learning_rate: learning_rate})
            for k, v in G.var.iteritems():
                self.params[k] = G.var[k].eval()
        self.updated = True

    def iterate_dataset(self, v, batch_size):
        """
        :param v: 2d np.array, each row stores an instance
        :return:
        """
        i = 0
        n_samples = v.shape[0]
        indices = range(n_samples)
        while True:
            if i + batch_size > n_samples:
                indices = np.random.permutation(n_samples)
                i = 0
            selected = indices[i : i + batch_size]
            i += batch_size
            yield v[selected]

    def __calc_free_energy__(self, V, W, b, c):
        """
        :param V: 2d tensor, (N, n_visible), each row stores an instance
        :param W: 2d tensor, (n_visible, n_hidden)
        :param b: 1d tensor, (n_visible, )
        :param c: 1d tensor, (n_hidden, )
        :return: the free energy of each sample, 1d tensor, (N,)
        """
        return -tf.reshape(tf.matmul(V, tf.reshape(b, [-1, 1])), [-1]) - tf.reduce_sum(tf.log(1 + tf.exp(c + tf.matmul(V, W))), reduction_indices=1)

    def gibbs_v(self, v0, W, b, c, k=1):
        """
        :param v0: 2d np.ndarray, (N, n_visible)
        :param W: 2d np.nadarray, (n_visible, n_hidden)
        :param b: 1d np.ndarray, (n_visible, )
        :param c: 1d np.ndarray, (n_hidden, )
        :param k:
        :return:
        """
        v = v0
        for i in range(k):
            h = self.sample_h_given_v(v, W, c)
            v = self.sample_v_given_h(h, W, b)
        return v

    def sample_h_given_v(self, v, W, c):
        """
        :param v: 2d np.ndarray, (N, n_visible)
        :param W: 2d np.nadarray, (n_visible, n_hidden)
        :param c: 1d np.ndarray, (n_hidden, )
        :return:
        """
        proba = self.sigmoid(np.matmul(v, W) + c)
        return self.sample_binomial(proba)

    def sample_v_given_h(self, h, W, b):
        """
        :param v: 2d np.ndarray, (N, n_visible)
        :param W: 2d np.nadarray, (n_visible, n_hidden)
        :param b: 1d np.ndarray, (n_visible, )
        :return:
        """
        proba = self.sigmoid(np.matmul(h, W.transpose()) + b)
        return self.sample_binomial(proba)

    def sample_binomial(self, proba):
        """
        :param proba: success probability, np.ndarray
        :return:
        """
        return np.float32(np.random.uniform(low=0.0, high=1.0, size=proba.shape) < proba)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


if __name__ == "__main__":
    plt.close('all')
    np.random.seed(1)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    n_hidden = 500
    learning_rate = 1e-3
    gibbs_steps = 10
    batch_size = 100
    num_epochs = 2000
    probe_epochs = 50
    rbm = RBM()
    train_v = np.float32(mnist.train.images > 0)
    validation_v = np.float32(mnist.validation.images[np.random.permutation(mnist.validation.images.shape[0])][0:1000] > 0)
    rbm.fit(train_v, validation_v=validation_v, n_hidden=n_hidden, gibbs_steps=gibbs_steps, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate, probe_epochs=probe_epochs)
    
    # sampling from the learnt distribution, starting from real samples
    gibbs_steps = 1
    x = np.float32(mnist.test.images[0:100, :] > 0)
    image = tile_raster_images(x, (28, 28), (10, 10))
    image = np.stack((image, image, image), axis=2)
    fig = plt.figure(0)
    ax = fig.add_subplot(121)
    ax.imshow(image)
    ax.axis('off')
    np.random.seed(1)
    x_sampling = rbm.gibbs_v(x, rbm.params['W'], rbm.params['b'], rbm.params['c'], k=gibbs_steps)
    image_sampling = Image.fromarray(tile_raster_images(x_sampling, (28, 28), (10, 10)))
    image_sampling = np.stack((image_sampling, image_sampling, image_sampling), axis=2)
    ax = fig.add_subplot(122)
    ax.imshow(image_sampling)
    ax.axis('off')
    ax.set_title('gibbs steps {s}'.format(s=gibbs_steps))
    fig.show()
    
    # sampling from the learnt distribution, starting from randoms
    probe_steps = 100
    v_sampling = np.zeros((100, 28*28), np.float32)
    np.random.seed(1)
    v0 = np.float32(np.random.random((1, 28*28)) > 0.5)
    v = v0
    for i in range(probe_steps * 100):
        if i % probe_steps == 0:
            v_sampling[int(i/probe_steps), :] = v
        h = rbm.sample_h_given_v(v, rbm.params['W'], rbm.params['c'])
        v = rbm.sample_v_given_h(h, rbm.params['W'], rbm.params['b'])
    image_sampling = tile_raster_images(v_sampling, (28, 28), (10, 10))
    image_sampling = np.stack((image_sampling, image_sampling, image_sampling), axis=2)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.imshow(image_sampling)
    ax.axis("off")
    fig.show()