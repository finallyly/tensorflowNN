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
    images = mnist.train.images
    labels = mnist.train.labels
    m = labels == 1
    labels = labels[m]
    images = images[m]
    
    idx = 6
    im = images[idx, :].reshape((28, 28))
    im = (im[0::3, 0::3] > 0.1).astype(int)
    x = im.reshape((-1))
    non_zeros = x.nonzero()[0]
    
    im_vis = 255 * im.astype(np.uint8)
    im_vis = np.stack((im_vis, im_vis, im_vis), axis=2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im_vis)
    ax.set_title(str(labels[idx]))
    ax.axis('off')
    fig.show()