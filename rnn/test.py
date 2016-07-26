# coding=utf-8
import tensorflow as tf
import numpy as np
import cPickle
import copy
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def generate_image_seq(image_set):
    seq = []
    for i in range(0, 10):
        idx = np.random.randint(0, image_set[i].shape[0])
        im = image_set[i][idx].reshape((28, 28))
        im = im[::2, ::2]
        im = (im > 0.1).astype(np.float32)
        im = im.reshape((-1))
        seq.append(im)
    return seq


if __name__ == '__main__':
    plt.close('all')

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    images = mnist.train.images
    labels = mnist.train.labels
    #m = labels == 6
    #labels = labels[m]
    #images = images[m]
    #
    #idx = 6
    #im = images[idx, :].reshape((28, 28))
    ##im = (im > 0.1).astype(int)
    #im = (im[0::2, 0::2] > 0.1).astype(int)
    #x = im.reshape((-1))
    #non_zeros = x.nonzero()[0]
    #
    #im_vis = 255 * im.astype(np.uint8)
    #im_vis = np.stack((im_vis, im_vis, im_vis), axis=2)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.imshow(im_vis)
    #ax.set_title(str(labels[idx]))
    #ax.axis('off')
    #fig.show()
    
    image_set = {}
    for l in range(0, 10):
        image_set[l] = images[labels == l]
    
    np.random.seed(0)
    seq = generate_image_seq(image_set)
    fig = plt.figure()
    i = 0
    for im in seq:
        im = im.reshape((14, 14))
        im_vis = 255 * im.astype(np.uint8)
        im_vis = np.stack((im_vis, im_vis, im_vis), axis=2)
        i += 1
        ax = fig.add_subplot(2,5, i)
        ax.axis('off')
        ax.imshow(im_vis)
    fig.show()
        
    #fig = plt.figure()
    #for i in range(0, 10):
    #    idx = np.random.randint(0, image_set[i].shape[0])
    #    im = image_set[i][idx].reshape((28, 28))
    #    im = im[::2, ::2]
    #    im = (im > 0.1).astype(np.float32)
    #    im_vis = 255 * im.astype(np.uint8)
    #    im_vis = np.stack((im_vis, im_vis, im_vis), axis=2)
    #    ax = fig.add_subplot(2,5, i+1)
    #    ax.axis('off')
    #    ax.imshow(im_vis)
    #fig.show()
            