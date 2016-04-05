# coding=utf-8
import tensorflow as tf
import numpy as np
import cPickle
import copy
import Image
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from utils import tile_raster_images

# def sample_binomial(proba, seed=None):
#     """
#     :param proba: success probability, tensor
#     :return: tensor
#     """
#     return tf.cast(tf.random_uniform(shape=proba.get_shape(), minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed) <= proba, tf.float32)


def sample_binomial(proba, seed=None):
    """
    :param proba: success probability, np.ndarray
    :return:
    """
    return np.float32(np.random.uniform(low=0.0, high=1.0, size=proba.shape) < proba)


def calc_free_energy(V, W, b, c):
    """
    :param V: 2d tensor, (N, n_visible), each row stores an instance
    :param W:
    :param b:
    :param c:
    :return: the free energy of each sample, 1d tensor, (N,)
    """
    return -tf.reshape(tf.matmul(V, tf.reshape(b, [-1, 1])), [-1]) - tf.reduce_sum(tf.log(1 + tf.exp(c + tf.matmul(V, W))), reduction_indices=1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sample_h_given_v(v, W, c):
    """
    :param v: 2d np.ndarray, (N, n_visible)
    :param W: 2d np.nadarray, (n_visible, n_hidden)
    :param c: 1d np.ndarray, (n_hidden, )
    :return:
    """
    proba = sigmoid(np.matmul(v, W) + c)
    return sample_binomial(proba)


def sample_v_given_h(h, W, b):
    proba = sigmoid(np.matmul(h, W.transpose()) + b)
    return sample_binomial(proba)


def gibbs_v(v0, W, b, c, k=1):
    v = v0
    for i in range(k):
        h = sample_h_given_v(v, W, c)
        v = sample_v_given_h(h, W, b)
    return v


if __name__ == '__main__':
    plt.close('all')
    np.random.seed(1121)

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)    
    n_visible = 28 * 28
    n_hidden = 500
    batch_size = 100
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
    loss = tf.reduce_mean(calc_free_energy(v, W, b, c)) - tf.reduce_mean(calc_free_energy(v_sampling, W, b, c))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for i in range(1001):
        batch_v, _ = mnist.train.next_batch(batch_size)
        batch_v = np.float32(batch_v > 0)
        batch_v_sampling = gibbs_v(batch_v, sess.run(W), sess.run(b), sess.run(c), k=15)
        
        if i % 50 == 0:
            loss_this_batch = sess.run(loss, feed_dict={v: batch_v, v_sampling: batch_v_sampling})
            reconstruct_err = np.mean(np.abs(batch_v - batch_v_sampling))
            print 'step {i}, loss {l:.4f}, reconstruction err {r:.6f}'.format(i=i, l=loss_this_batch, r=reconstruct_err)
        
            x = np.float32(mnist.test.images[0:100, :] > 0)
            image = tile_raster_images(x, (28, 28), (10, 10))
            image = np.stack((image, image, image), axis=2)
            fig = plt.figure(0)
            ax = fig.add_subplot(121)
            ax.imshow(image)
            ax.axis('off')
            x_sampling = gibbs_v(x, sess.run(W), sess.run(b), sess.run(c), k=1)
            image_sampling = Image.fromarray(tile_raster_images(x_sampling, (28, 28), (10, 10)))
            image_sampling = np.stack((image_sampling, image_sampling, image_sampling), axis=2)
            ax = fig.add_subplot(122)
            ax.imshow(image_sampling)
            ax.axis('off')
            fig.savefig("results/step{i}.png".format(i=i))
        
        sess.run(train_step, feed_dict={v: batch_v, v_sampling: batch_v_sampling})