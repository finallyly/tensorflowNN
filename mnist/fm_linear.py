#coding=utf-8
import numpy as np
import tensorflow as tf
import cPickle
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.examples.tutorials.mnist import input_data

def calc_XXT(X):
    return np.apply_along_axis(lambda x: np.outer(x, x).reshape((-1)), 1, X)

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    m = (mnist.train.labels == 2) | ((mnist.train.labels == 5))
    train_images = mnist.train.images[m]
    train_labels = np.float32(mnist.train.labels[m])
    train_labels[train_labels == 2] = -1
    train_labels[train_labels == 5] = 1

    m = (mnist.test.labels == 2) | ((mnist.test.labels == 5))
    test_images = mnist.test.images[m]
    test_labels = np.float32(mnist.test.labels[m])
    test_labels[test_labels == 2] = -1
    test_labels[test_labels == 5] = 1
    
    input_dim = train_images.shape[1]
    latent_dim = 10
    learning_rate = 1e-2
    batch_size = 50
    num_epochs = 1000
    
    # build graph
    X = tf.placeholder(tf.float32, (None, input_dim))
    y_ = tf.placeholder(tf.float32, (None,))
    w0 = tf.Variable(0.0)
    w = tf.Variable(tf.zeros([input_dim, 1]))
    y = w0 + tf.matmul(X, w)
    y = tf.reshape(y, [-1])
    nll = -tf.reduce_mean(tf.log(tf.nn.sigmoid(y * y_)))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(nll)
    is_correct = tf.greater(y * y_, 0)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    # train model
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    num_samples = train_images.shape[0]
    head = 0
    indices = range(num_samples)
    for i in range(num_epochs):
        if head + batch_size > num_samples:
            indices = np.random.permutation(num_samples)
            head = 0
        selected = indices[head: head + batch_size]
        head += batch_size
        batch_x = train_images[selected]
        batch_y = train_labels[selected]
        if i%100 == 0:
            accuracy_rate = sess.run(accuracy, feed_dict={X: batch_x, y_: batch_y})
            print 'step {i}, accuracy on the batch is {a:.2f}%'.format(i=i, a=accuracy_rate*100.0)
        sess.run(train_step, feed_dict={X: batch_x, y_: batch_y})
    w0_ = sess.run(w0)
    w_ = sess.run(w)
    # test model
    accuracy_rate = sess.run(accuracy, feed_dict={X: test_images, y_: test_labels})
    print 'accuracy on test set is {a:.2f}%'.format(a=accuracy_rate*100.0)
    #sess.close()
    