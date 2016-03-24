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
    m = (mnist.train.labels == 0) | ((mnist.train.labels == 8))
    train_images = mnist.train.images[m]
    train_labels = np.float32(mnist.train.labels[m])
    train_labels[train_labels == 0] = -1
    train_labels[train_labels == 8] = 1
    train_images = train_images[0:1000]
    train_labels = train_labels[0:1000]
    
    input_dim = train_images.shape[1]
    latent_dim = 10
    beta1 = 0.1
    beta2 = 0.1
    learning_rate = 1e-4
    batch_size = 50
    num_epochs = 10
    
    # build graph
    X = tf.placeholder(tf.float32, (None, input_dim))
    XXT = tf.placeholder(tf.float32, (None, input_dim * input_dim))
    y_ = tf.placeholder(tf.float32, (None,))
    w0 = tf.Variable(0.0)
    w = tf.Variable(tf.zeros([input_dim, 1]))
    V = tf.Variable(tf.truncated_normal([input_dim, latent_dim], stddev=1.0/np.sqrt(input_dim), seed=0))
    #V = tf.Variable(tf.zeros([input_dim, latent_dim]))
    VVT = tf.matmul(V, V, transpose_a=False, transpose_b=True)
    y = w0 + tf.matmul(X, w) + tf.matmul(XXT, tf.reshape(VVT, [-1, 1]))
    y = tf.reshape(y, [-1])
    nll = -tf.reduce_mean(tf.log(tf.nn.sigmoid(y * y_)))
    #loss = nll + beta1 * tf.reduce_sum(w * w) + beta2 * 
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(nll)
    is_correct = tf.greater(y * y_, 0)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    # train model
    train_images_xxt = calc_XXT(train_images)
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
        batch_xxt = train_images_xxt[selected]
        batch_y = train_labels[selected]
        #if i%probe_step == 0:
        accuracy_rate = sess.run(accuracy, feed_dict={X: batch_x, XXT: batch_xxt, y_: batch_y})
        print 'step {i}, accuracy on the batch is {a:.2f}'.format(i=i, a=accuracy_rate*100.0)
        sess.run(train_step, feed_dict={X: batch_x, XXT: batch_xxt, y_: batch_y})
    w0_ = sess.run(w0)
    w_ = sess.run(w)
    V_ = sess.run(V)
    #sess.close()
    
    # test model
