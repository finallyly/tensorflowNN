# coding=utf-8
import tensorflow as tf
import numpy as np

class Struct(dict):
    def __init__(self, **kwargs):
        super(Struct, self).__init__(**kwargs)
        self.__dict__ = self


class GraphWrapper(object):
    def __init__(self, graph, phr, var, tsr, ops):
        self.graph = graph
        self.phr = phr
        self.var = var
        self.tsr = tsr
        self.ops = ops


def iterate_dataset(x, y=None, batch_size=50, seed=None):
    """
    generate a iterator over the given dataset
    :param v: 2d np.array, each row stores an instance
    :param y: 1d np.array, labels/targets
    :param batch_size:
    :return: a batch of (v, y)
    """
    if seed is not None:
        np.random.seed(seed)
    i = 0
    n_samples = x.shape[0]
    if y is not None and y.shape[0] != n_samples:
        raise Exception("iterate_dataset: x and y have conflicted sizes")
    indices = np.random.permutation(n_samples)
    while True:
        if i + batch_size > n_samples:
            indices = np.random.permutation(n_samples)
            i = 0
        selected = indices[i : i + batch_size]
        i += batch_size
        if y is not None:
            yield x[selected], y[selected]
        else:
            yield x[selected]

def ordinal_to_onehot(y, nbits=None):
    """
    :param y: 1d np.ndarray, starting from zero
    :param nbits:
    :return: 2d np.ndarray
    """
    if nbits is None:
        nbits = y.max() + 1
    yy = np.zeros((y.shape[0], nbits), dtype=np.int32)
    yy[np.arange(y.shape[0]), y] = 1
    return yy


def sample_binomial(proba):
    """
    :param proba: success probability, 1d np.ndarray
    :return:
    """
    return np.float32(np.random.uniform(low=0.0, high=1.0, size=proba.shape) < proba)


def sigmoid(x):
    """
    :param x: 1d np.array or floating scalar
    :return:
    """
    return 1.0 / (1.0 + np.exp(-x))