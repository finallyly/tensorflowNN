# coding=utf-8
import tensorflow as tf
import numpy as np
import cPickle
import copy
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    images = mnist.train.images
    labels = mnist.train.labels
    dataset = []
    for i in range(0, images.shape[0]):
        if labels[i] == 1:
            im = images[i, :].reshape((28, 28))
            im = (im[0::3, 0::3] > 0.1).astype(int)
            # image中非零元素的index
            nonzeros = im.reshape((-1)).nonzero()[0]
            dataset.append((nonzeros, labels[i]))
            if i % 1000 == 0:
                print '{i} samples processed'.format(i=i)
    cPickle.dump(dataset, open("MNIST_data/mnist_seq.pkl", "wb"))
