# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
import cPickle
from tensorflow.examples.tutorials.mnist import input_data


def generate_image_seq(image_set):
    seq = []
    for i in range(0, 10):
        idx = np.random.randint(0, image_set[i].shape[0])
        im = image_set[i][idx].reshape((28, 28))
        im = im[::2, ::2]
        im = (im > 0.1).astype(np.float32)
        im = im.reshape((-1))
        seq.append(im)
    return seq


class RNN(object):
    def __init__(self, input_size, hidden_size, output_size, num_steps, learning_rate):
        with tf.variable_scope('rnn', reuse=False):
            U = tf.get_variable('U', shape=[hidden_size, input_size], dtype=tf.float32, initializer=tf.uniform_unit_scaling_initializer())
            W = tf.get_variable('W', shape=[hidden_size, hidden_size], dtype=tf.float32, initializer=tf.uniform_unit_scaling_initializer())
            V = tf.get_variable('V', shape=[output_size, hidden_size], dtype=tf.float32, initializer=tf.uniform_unit_scaling_initializer())
            b = tf.get_variable('b', shape=[hidden_size], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            c = tf.get_variable('c', shape=[output_size], dtype=tf.float32, initializer=tf.constant_initializer(0.))
        data = tf.placeholder(dtype=tf.float32, shape=[num_steps, input_size], name='input')
        target = tf.placeholder(dtype=tf.float32, shape=[num_steps, input_size], name='target')
        s = tf.constant(0., dtype=tf.float32, shape=[hidden_size])
        loss = 0
        for t in range(0, num_steps):
            x = data[t, :]
            y = target[t, :]
            Ux = tf.matmul(U, tf.reshape(x, [-1, 1]))
            Ux = tf.reshape(Ux, [-1])
            Ws = tf.matmul(W, tf.reshape(s, [-1, 1]))
            Ws = tf.reshape(Ws, [-1])
            s = tf.tanh(Ux + Ws + b)
            Vs = tf.matmul(V, tf.reshape(s, [-1, 1]))
            Vs = tf.reshape(Vs, [-1])
            z = tf.nn.sigmoid(Vs + c)
            loss += tf.reduce_sum(-tf.log(z) * y)
            # TODO: why this loss does not work
            #loss += tf.reduce_sum(tf.log(z) * (1 - 2*y))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        self.data = data
        self.target = target
        self.loss = loss
        self.train_step = train_step
        self.vars = [U, W, V, b, c]


if __name__ == '__main__':
    np.random.seed(0)
    
    num_samples = 1000000
    input_size = 14*14
    output_size = input_size
    hidden_size = int(0.5 * input_size)
    num_steps = 9
    learning_rate = 1e-3
    
    rnn = RNN(input_size, hidden_size, output_size, num_steps, learning_rate)
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    images = mnist.train.images
    labels = mnist.train.labels
    image_set = {}
    for l in range(0, 10):
        image_set[l] = images[labels == l]
    
    valid_set = []
    for i in range(0, 100):
        valid_set.append(np.vstack(generate_image_seq(image_set)))
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for i in range(0, num_samples):
        # validate the model
        if i % 100 == 0:
            loss = []
            for seq in valid_set:
                data = seq[0:-1, :]
                target = seq[1:, :]
                loss.append(sess.run(rnn.loss, feed_dict={rnn.data: data, rnn.target: target}))
            loss = np.mean(loss)
            print '{n} samples processed, validation loss {l:.8f}'.format(l=loss, n=i)
        # dump the learnt parameters
        if i % 1000 == 0:
            var_list = []
            for var in rnn.vars:
                var_list.append(sess.run(var))
            cPickle.dump(var_list, open("model/rnn_sample{i}.pkl".format(i=i), "wb"))
            
        seq = generate_image_seq(image_set)
        seq = np.vstack(seq)
        data = seq[0:-1, :]
        target = seq[1:, :]
        sess.run(rnn.train_step, feed_dict={rnn.data: data, rnn.target: target})
    sess.close()