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
    print 'training set: {p} positives, {n} negatives'.format(p=(train_labels==1).sum(), n=(train_labels==-1).sum())
    
    m = (mnist.test.labels == 2) | ((mnist.test.labels == 5))
    test_images = mnist.test.images[m]
    test_labels = np.float32(mnist.test.labels[m])
    test_labels[test_labels == 2] = -1
    test_labels[test_labels == 5] = 1
    print 'test set: {p} positives, {n} negatives'.format(p=(test_labels==1).sum(), n=(test_labels==-1).sum())
    
    input_dim = train_images.shape[1]
    latent_dim = 10
    learning_rate = 1e-2
    penalty_V = 1e-2
    penalty_w = 1e-2
    batch_size = 50
    num_epochs = 200
    probe_step = 10
    
    # build graph
    X = tf.placeholder(tf.float32, (None, input_dim))
    XXT = tf.placeholder(tf.float32, (None, input_dim * input_dim))
    y_ = tf.placeholder(tf.float32, (None,))
    w0 = tf.Variable(0.0)
    w = tf.Variable(tf.truncated_normal([input_dim, 1], stddev=1.0/np.sqrt(input_dim), seed=0))
    V = tf.Variable(tf.truncated_normal([input_dim, latent_dim], stddev=1.0/np.sqrt(input_dim), seed=1))
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
    
    # train model
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    print '|V|.sum is {s}'.format(s=np.abs(sess.run(V)).sum())
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
        batch_xxt = calc_XXT(batch_x)
        if i%probe_step == 0:
            nll_this_step = sess.run(nll, feed_dict={X: batch_x, XXT: batch_xxt, y_: batch_y})
            reg_V_this_step = sess.run(reg_V)
            reg_w_this_step = sess.run(reg_w)
            accuracy_this_step = sess.run(accuracy, feed_dict={X: batch_x, XXT: batch_xxt, y_: batch_y})
            print 'step {i}, accuracy {a:.2f}%, reg_V {v:.3f}, reg_w {w:.3f}, nll {n:.3f}'\
                .format(i=i, a=accuracy_this_step*100.0, v=reg_V_this_step, w=reg_w_this_step, n=nll_this_step)
        sess.run(train_step, feed_dict={X: batch_x, XXT: batch_xxt, y_: batch_y})
    # test model
    predictions = None
    num_test_samples = test_images.shape[0]
    test_batch = 50
    for i in range(0, int(num_test_samples/batch_size)):
        batch_x = test_images[i*test_batch : (i+1)*test_batch]
        batch_xxt = calc_XXT(batch_x)
        pred = sess.run(y, feed_dict={X: batch_x, XXT: batch_xxt})
        if predictions is None:
            predictions = pred
        else:
            predictions = np.concatenate((predictions, pred))
    n = predictions.shape[0]
    targets = test_labels[0:n]
    accuracy_test_set = (predictions * targets > 0).mean()
    print 'accuracy on test set is {a:.2f}%'.format(a=accuracy_test_set*100.0)
    #sess.close()