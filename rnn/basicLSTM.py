# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
import cPickle


class BasicLSTM(object):
    def __init__(self, hidden_size, input_size, output_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        with tf.variable_scope(type(self).__name__, reuse=False):
            # TODO: use tensor array
            # forget gate
            Wf = tf.get_variable("Wf", shape=(self.hidden_size, self.hidden_size + self.input_size),
                                 dtype=np.float32, initializer=tf.uniform_unit_scaling_initializer())
            bf = tf.get_variable("bf", shape=self.hidden_size, dtype=np.float32,
                                 initializer=tf.constant_initializer(0.0))
            # input gate
            Wi = tf.get_variable("Wi", shape=(self.hidden_size, self.hidden_size + self.input_size),
                                 dtype=np.float32, initializer=tf.uniform_unit_scaling_initializer())
            bi = tf.get_variable("bi", shape=self.hidden_size, dtype=np.float32,
                                 initializer=tf.constant_initializer(0.0))
            # output gate
            Wo = tf.get_variable("Wo", shape=(self.hidden_size, self.hidden_size + self.input_size),
                                 dtype=np.float32, initializer=tf.uniform_unit_scaling_initializer())
            bo = tf.get_variable("bo", shape=self.hidden_size, dtype=np.float32,
                                 initializer=tf.constant_initializer(0.0))
            # new state candidates
            Wc = tf.get_variable("Wc", shape=(self.hidden_size, self.hidden_size + self.input_size),
                                 dtype=np.float32, initializer=tf.uniform_unit_scaling_initializer())
            bc = tf.get_variable("bc", shape=self.hidden_size, dtype=np.float32,
                                 initializer=tf.constant_initializer(0.0))
            # output matrix
            Wout = tf.get_variable("Wout", shape=(self.output_size, self.hidden_size),
                                 dtype=np.float32, initializer=tf.uniform_unit_scaling_initializer())
            bout = tf.get_variable("bout", shape=self.output_size, dtype=np.float32,
                                 initializer=tf.constant_initializer(0.0))

    def __call__(self, inputs, targets):
        """

        :param inputs: 2d np.array, each row is an instance
        :param targets: 2d np.array, each row is a onehot label
        :return:
        """
        num_steps = inputs.shape[0]
        assert(inputs.shape[1] == self.input_size)
        h = tf.constant(np.zeros(self.hidden_size), dtype=tf.float32)
        c = tf.constant(np.zeros(self.hidden_size), dtype=tf.float32)
        loss = 0
        for t in range(0, num_steps):
            #TODO: dig into the use of variable scope
            with tf.variable_scope(type(self).__name__, reuse=True):
                Wf = tf.get_variable("Wf")
                bf = tf.get_variable("bf")
                Wi = tf.get_variable("Wi")
                bi = tf.get_variable("bi")
                Wo = tf.get_variable("Wo")
                bo = tf.get_variable("bo")
                Wc = tf.get_variable("Wc")
                bc = tf.get_variable("bc")
                Wout = tf.get_variable("Wout")
                bout = tf.get_variable("bout")
            x = inputs[t, :]
            y = targets[t]
            h_x = tf.reshape(array_ops.concat(0, (h, x)), [-1, 1])
            # forget gate
            f = tf.sigmoid(tf.reshape(tf.matmul(Wf, h_x), [-1]) + bf)
            # input gate
            i = tf.sigmoid(tf.reshape(tf.matmul(Wi, h_x), [-1]) + bi)
            # output gate
            o = tf.sigmoid(tf.reshape(tf.matmul(Wo, h_x), [-1]) + bo)
            # new state candidates
            cand_c = tf.tanh(tf.reshape(tf.matmul(Wc, h_x), [-1]) + bc)
            # new state
            c = f * c + i * cand_c
            # new hidden state
            h = o * tf.tanh(c)
            output = tf.reshape(tf.matmul(Wout, tf.reshape(h, [-1, 1])), [-1]) + bout
            output = tf.reshape(tf.nn.softmax(tf.reshape(output, [1, -1])), [-1])
            loss += -tf.reduce_sum(tf.log(output) * y)
        return loss, output
        

if __name__ == '__main__':
    dataset = cPickle.load(open('MNIST_data/mnist_seq.pkl', 'rb'))

    output_size = input_size = 196
    hidden_size = 100
    lstm_cell = BasicLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

    x = dataset[0][0]
    X = np.zeros((len(x), input_size), np.float32)
    X[np.arange(0, X.shape[0]), x] = 1.0
    data = X[0:-1, :]
    target = X[1:, :]
    loss = lstm_cell(data, target)
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    l, out = sess.run(loss)
    