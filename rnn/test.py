# coding=utf-8
import tensorflow as tf
import numpy as np
import cPickle
import copy
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    plt.close('all')

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    x = mnist.train.images
    y = mnist.train.labels
    im = x[0, :].reshape((28, 28))
    im = 255 * (im > 0).astype(np.uint8)
    im = np.stack((im, im, im), axis=2)
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    ax.imshow(im)
    ax.axis('off')
    fig.show()