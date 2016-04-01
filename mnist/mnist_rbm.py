# coding=utf-8
import tensorflow as tf
import numpy as np

# def sample_binomial(proba, seed=None):
#     """
#     :param proba: success probability, tensor
#     :return: tensor
#     """
#     return tf.cast(tf.random_uniform(shape=proba.get_shape(), minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed) <= proba, tf.float32)


def sample_binomial(proba, seed=None):
    """
    :param proba: success probability, 1d np.ndarray
    :return: 1d np.ndarray
    """
    return np.float32(np.random.uniform(low=0.0, high=1.0, size=proba.shape) < proba)


def calc_free_energy(V, W, b, c):
    return -tf.reshape(tf.matmul(V, tf.reshape(b, [-1, 1])), [-1]) - tf.reduce_sum(tf.log(1 + tf.exp(c + tf.matmul(V, W))), reduction_indices=1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sample_h_given_v(v, W, c):
    """
    :param v: 1d np.ndarray, (n_visible, )
    :param W: 2d np.nadarray, (n_visible, n_hidden)
    :param c: 1d np.ndarray, (n_hidden, )
    :return:
    """
    proba = sigmoid(np.matmul(W.transpose(), v) + c)
    return sample_binomial(proba)


def sample_v_given_h(h, W, b):
    proba = sigmoid(np.matmul(W, h) + b)
    return sample_binomial(proba)


if __name__ == '__main__':
    n_visible = 10
    n_hidden = 5
    initial_W = np.random.uniform(
        low=-4 * np.sqrt(6.0 / (n_hidden + n_visible)),
        high=4 * np.sqrt(6.0 / (n_hidden + n_visible)),
        size=(n_visible, n_hidden)
    )
    W = tf.Variable(initial_value=initial_W, trainable=True)
    b = tf.Variable(np.zeros((n_visible,), np.float32))
    c = tf.Variable(np.zeros((n_hidden,), np.float32))
    V = tf.placeholder(tf.float32, [None, n_visible])
