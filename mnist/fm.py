#coding=utf-8
import numpy as np
import tensorflow as tf
import cPickle
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    m = (mnist.train.labels == 0) | ((mnist.train.labels == 8))
    train_images = mnist.train.images[m]
    train_labels = mnist.train.labels[m]
    
    m = (mnist.validation.labels == 0) | ((mnist.validation.labels == 8))
    validation_images = mnist.validation.images[m]
    validation_labels = mnist.validation.labels[m]
    
    m = (mnist.test.labels == 0) | ((mnist.test.labels == 8))
    test_images = mnist.test.images[m]
    test_labels = mnist.test.labels[m]