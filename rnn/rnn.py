# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
import cPickle


class RNN(object):
    def __init__(self, input_size, hidden_size, output_size, num_steps):
        with tf.variable_scope('rnn', reuse=False):
            U = tf.get_variable('U', shape=[hidden_size, input_size], dtype=tf.float32, initializer=tf.uniform_unit_scaling_initializer())
            W = tf.get_variable('W', shape=[hidden_size, hidden_size], dtype=tf.float32, initializer=tf.uniform_unit_scaling_initializer())
            V = tf.get_variable('V', shape=[output_size, hidden_size], dtype=tf.float32, initializer=tf.uniform_unit_scaling_initializer())
            b = tf.get_variable('b', shape=[hidden_size], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            c = tf.get_variable('c', shape=[output_size], dtype=tf.float32, initializer=tf.constant_initializer(0.))
        data = tf.placeholder(dtype=tf.float32, shape=[num_steps, input_size], name='input')
        target = tf.placeholder(dtype=tf.float32, shape=[num_steps, input_size], name='target')
        target_weight = tf.placeholder(dtype=tf.float32, shape=[num_steps], name='target_weight')
        learning_rate = tf.placeholder(dtype=tf.float32)
        s = tf.constant(0., dtype=tf.float32, shape=[hidden_size])
        loss = 0
        for t in range(0, num_steps):
            x = data[t, :]
            y = target[t, :]
            weight = target_weight[t]
            Ux = tf.matmul(U, tf.reshape(x, [-1, 1]))
            Ux = tf.reshape(Ux, [-1])
            Ws = tf.matmul(W, tf.reshape(s, [-1, 1]))
            Ws = tf.reshape(Ws, [-1])
            s = tf.tanh(Ux + Ws + b)
            Vs = tf.matmul(V, tf.reshape(s, [-1, 1]))
            Vs = tf.reshape(Vs, [-1])
            z = tf.nn.softmax(tf.reshape(Vs + c, [1, -1]))
            z = tf.reshape(z, [-1])
            loss += -tf.reduce_sum(tf.log(z) * y) * weight
        loss /= tf.reduce_sum(target_weight)
        # train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        self.data = data
        self.target = target
        self.target_weight = target_weight
        self.loss = loss
        self.train_step = train_step
        self.vars = [U, W, V, b, c]
        self.learning_rate = learning_rate


if __name__ == '__main__':
    np.random.seed(0)
    
    dataset = cPickle.load(open('English_words/dataset.pkl', 'rb'))
    
    output_size = input_size = 28 # including eos and padding
    eos = 26
    padding = 27
    seq_size = 11 # incl. eos
    num_steps = seq_size - 1
    hidden_size = input_size
    learning_rate = 0.1
    
    num_epochs = 10
    
    rnn = RNN(input_size, hidden_size, output_size, num_steps)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for ep in range(0, num_epochs):
        i = 0
        avg_loss = 0
        for data, target, target_weight in dataset:
            i += 1
            _, loss = sess.run([rnn.train_step, rnn.loss],
                               feed_dict={rnn.data: data, rnn.target: target, rnn.target_weight: target_weight, rnn.learning_rate: learning_rate})
            avg_loss += loss
            if i % 1000 == 0:
                avg_loss /= 1000
                print 'epoch {ep}, {i}-th sample, avg loss {l}, lr {lr}'\
                    .format(ep=ep, i=i, l=avg_loss, lr=learning_rate)
                avg_loss = 0
        learning_rate /= 2
    sess.close()