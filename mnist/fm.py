#coding=utf-8
import numpy as np
import tensorflow as tf
import cPickle
from tensorflow.examples.tutorials.mnist import input_data

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


class FactorMach(object):
    def __init__(self):
        # 模型的参数,对应于Graph中的trainable variables
        self.params = {}
        # Graph and session for online running
        self.updated = False
        self.G_run = None
        self.sess_run = None
        
    def __del__(self):
        if self.sess_run is not None:
            self.sess_run.close()

    def __build_graph__(self, input_dim=None, latent_dim=None, var_val=None):
        graph = tf.Graph()
        with graph.as_default():
            if var_val is not None:
                input_dim = var_val['V'].shape[0]
                latent_dim = var_val['V'].shape[1]
                w0 = tf.Variable(var_val['w0'])
                w = tf.Variable(var_val['w'])
                V = tf.Variable(var_val['V'])
            else:
                w0 = tf.Variable(0.0)
                w = tf.Variable(tf.truncated_normal([input_dim, 1], stddev=1.0/np.sqrt(input_dim), seed=0))
                V = tf.Variable(tf.truncated_normal([input_dim, latent_dim], stddev=1.0/np.sqrt(input_dim), seed=1))
            X = tf.placeholder(tf.float32, (None, input_dim))
            XXT = tf.placeholder(tf.float32, (None, input_dim * input_dim))
            y_ = tf.placeholder(tf.float32, (None,))
            learning_rate = tf.placeholder(tf.float32)
            penalty_w = tf.placeholder(tf.float32)
            penalty_V = tf.placeholder(tf.float32)
            VVT = tf.matmul(V, V, transpose_a=False, transpose_b=True)
            y = w0 + tf.matmul(X, w) + tf.matmul(XXT, tf.reshape(VVT, [-1, 1]))
            y = tf.reshape(y, [-1])
            nll = -tf.reduce_mean(tf.log(tf.nn.sigmoid(y * y_)))
            reg_V = tf.reduce_sum(V * V)
            reg_w = tf.reduce_sum(w * w)
            loss = nll + penalty_w * reg_w + penalty_V * reg_V
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            is_correct = tf.greater(y * y_, 0)
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
            init_vars = tf.initialize_all_variables()
        phr = Struct(X=X, XXT=XXT, y_=y_, learning_rate=learning_rate, penalty_w=penalty_w, penalty_V=penalty_V)
        var = Struct(w0=w0, w=w, V=V)
        tsr = Struct(y=y, nll=nll, accuracy=accuracy)
        ops = Struct(train_step=train_step, init_vars=init_vars)
        return GraphWrapper(graph, phr, var, tsr, ops)

    def fit(self, x, y, latent_dim, batch_size, num_epochs, learning_rate=1e-3, penalty_w=1e-2, penalty_V=1e-2, verbose=True, probe_epochs=100):
        """
        :param x:
        :param y: numpy vector, {-1, 1}
        :param learning_rate:
        :param penalty_w:
        :param penalty_V:
        :param batch_size:
        :param num_epochs:
        :param verbose:
        :return:
        """
        unique_labels = np.unique(y)
        if not (len(unique_labels) == 2 and unique_labels[0] == -1 and unique_labels[1] == 1):
            raise Exception("labels must be {-1, 1}")
        num_samples = x.shape[0]
        input_dim = x.shape[1]
        G = self.__build_graph__(input_dim, latent_dim)
        sess = tf.Session(graph=G.graph)
        with sess.as_default():
            G.ops.init_vars.run()
            head = 0
            indices = range(num_samples)
            for i in range(num_epochs):
                if head + batch_size > num_samples:
                    indices = np.random.permutation(num_samples)
                    head = 0
                selected = indices[head: head + batch_size]
                head += batch_size
                batch_x = x[selected]
                batch_xxt = self.__calc_XXT__(batch_x)
                batch_y = y[selected]
                if verbose:
                    if i % probe_epochs == 0:
                        accuracy = G.tsr.accuracy.eval(feed_dict={G.phr.X: batch_x, G.phr.XXT: batch_xxt, G.phr.y_: batch_y})
                        print 'step {s}, training batch, accuracy {a:.2f}%'.format(s=i, a=accuracy * 100.0)
                G.ops.train_step.run(feed_dict={G.phr.X: batch_x, G.phr.XXT: batch_xxt, G.phr.y_: batch_y,
                                                G.phr.learning_rate: learning_rate,
                                                G.phr.penalty_w: penalty_w, G.phr.penalty_V: penalty_V})
            for k, v in G.var.iteritems():
                self.params[k] = G.var[k].eval()
        self.updated = True

    def predict(self, x):
        if len(self.params) < 1:
            raise Exception("empty model")
        if self.updated:
            self.G_run = self.__build_graph__(var_val=self.params)
            if self.sess_run is not None: self.sess_run.close()
            self.sess_run = tf.Session(graph=self.G_run.graph)
            self.sess_run.run(self.G_run.ops.init_vars)
            self.updated = False
        num_samples = x.shape[0]
        batch_size = 50
        predictions = np.zeros((num_samples, ), dtype=np.float32)
        for i in range(0, int(np.ceil(1.0 * num_samples / batch_size))):
            a = i * batch_size
            b = min((i + 1) * batch_size, num_samples)
            batch_x = x[a : b]
            batch_xxt = self.__calc_XXT__(batch_x)
            y = self.sess_run.run(self.G_run.tsr.y, feed_dict={self.G_run.phr.X: batch_x, self.G_run.phr.XXT: batch_xxt})
            predictions[a: b] = y
        return predictions

    def save(self, path):
        if len(self.params) < 1:
            raise Exception("empty model")
        with open(path, "wb") as f:
            cPickle.dump(self.params, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.params = cPickle.load(f)
        self.updated = True

    def __calc_XXT__(self, X):
        return np.apply_along_axis(lambda x: np.outer(x, x).reshape((-1)), 1, X)


if __name__ == '__main__':
    digit1 = 5
    digit2 = 6
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    m = (mnist.train.labels == digit1) | ((mnist.train.labels == digit2))
    train_images = mnist.train.images[m]
    train_labels = np.float32(mnist.train.labels[m])
    train_labels[train_labels == digit1] = -1
    train_labels[train_labels == digit2] = 1
    print 'training set: {p} positives, {n} negatives'.format(p=(train_labels==1).sum(), n=(train_labels==-1).sum())
    
    m = (mnist.validation.labels == digit1) | ((mnist.validation.labels == digit2))
    validation_images = mnist.validation.images[m]
    validation_labels = np.float32(mnist.validation.labels[m])
    validation_labels[validation_labels == digit1] = -1
    validation_labels[validation_labels == digit2] = 1
    print 'validation set: {p} positives, {n} negatives'.format(p=(validation_labels==1).sum(), n=(validation_labels==-1).sum())
    
    m = (mnist.test.labels == digit1) | ((mnist.test.labels == digit2))
    test_images = mnist.test.images[m]
    test_labels = np.float32(mnist.test.labels[m])
    test_labels[test_labels == digit1] = -1
    test_labels[test_labels == digit2] = 1
    print 'test set: {p} positives, {n} negatives'.format(p=(test_labels==1).sum(), n=(test_labels==-1).sum())

    fm = FactorMach()
    fm.fit(train_images, train_labels,
           latent_dim=10, batch_size=50, num_epochs=100,
           learning_rate=1e-2, penalty_w=1e-2, penalty_V=1e-2,
           verbose=True, probe_epochs=10)
    predictions = fm.predict(test_images)
    accuracy = np.mean(predictions * test_labels > 0)
    print 'test set: accuracy {a:.2f}%'.format(a=accuracy*100.0)