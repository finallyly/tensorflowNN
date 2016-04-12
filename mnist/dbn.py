# coding=utf-8
import tensorflow as tf
import numpy as np
import cPickle
import copy
import Image
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from util import tile_raster_images
from tf_util import Struct, GraphWrapper
from rbm import RBM


class DBN(object):
    def __init__(self, n_hidden=[], gibbs_steps=[], batch_size=[], num_epochs=[], learning_rate=[], probe_epochs=[]):
        self.rbm_layers = []
        n_rbm = len(n_hidden)
        for i in range(n_rbm):
            rbm = RBM(n_hidden=n_hidden[i],
                      gibbs_steps=gibbs_steps[i],
                      batch_size=batch_size[i],
                      num_epochs=num_epochs[i],
                      learning_rate=learning_rate[i],
                      probe_epochs=probe_epochs[i])
            self.rbm_layers.append(rbm)

    def pretrain(self, v, validation_v=None):
        input = v
        validation_input = validation_v
        for rbm, i in zip(self.rbm_layers, range(len(self.rbm_layers))):
            print '### pretraining RBM Layer {i}'.format(i=i)
            rbm.fit(input, validation_input)
            output = rbm.sample_h_given_v(input, rbm.params['W'], rbm.params['c'])
            if validation_input is not None:
                validation_output = rbm.sample_h_given_v(validation_input, rbm.params['W'], rbm.params['c'])
            else:
                validation_output = None
            input = output
            validation_input = validation_output


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
    dbn = DBN(n_hidden=n_hidden,
              gibbs_steps=gibbs_steps,
              batch_size=batch_size,
              num_epochs=num_epochs,
              learning_rate=learning_rate,
              probe_epochs=probe_epochs)
    dbn.pretrain(train_v, validation_v)
