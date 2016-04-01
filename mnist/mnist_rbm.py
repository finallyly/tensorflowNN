# coding=utf-8
import tensorflow as tf

def sample_binomial(proba, seed=None):
    """
    :param proba: success probability, tensor
    :return: tensor
    """
    return tf.cast(tf.random_uniform(shape=proba.get_shape(), minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed) <= proba, tf.float32)
