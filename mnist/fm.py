#coding=utf-8
import numpy as np
import tensorflow as tf
import cPickle
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.examples.tutorials.mnist import input_data

def predict(x, w0, w, VVT):
    """
    :param x: tensor, an instance
    :param w0: scalar tensor
    :param w: variable tensor, vector
    :param V: variable tensor, matrix
    :return: tensor
    """
    x_ = tf.reshape(x, [-1, 1])
    y = w0 + tf.reduce_sum(x * w) \
        + tf.reduce_sum(tf.matmul(x_, x_, transpose_a=False, transpose_b=True) * VVT)
    return y

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    m = (mnist.train.labels == 0) | ((mnist.train.labels == 8))
    train_images = mnist.train.images[m]
    train_labels = np.float32(mnist.train.labels[m])
    train_labels[train_labels == 0] = -1
    train_labels[train_labels == 8] = 1
    
    num_samples = train_images.shape[0]
    input_dim = train_images.shape[1]
    latent_dim = 10

    X = tf.placeholder(tf.float32, (num_samples, input_dim))
    y_ = tf.placeholder(tf.float32, (num_samples,))
    w0 = tf.Variable(0.0)
    w = tf.Variable(tf.zeros([input_dim]))
    V = tf.Variable(tf.zeros([input_dim, latent_dim]))
    VVT = tf.matmul(V, V, transpose_a=False, transpose_b=True)
    y = tf.pack([predict(x, w0, w, VVT) for x in array_ops.unpack(X)])
    nll = -tf.reduce_mean(tf.nn.sigmoid(y * y_))
    is_correct = tf.greater(y * y_, 0)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


