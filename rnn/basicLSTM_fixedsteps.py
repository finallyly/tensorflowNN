# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
import cPickle


class BasicLSTM(object):
    def __init__(self, hidden_size, input_size, output_size, num_steps, learning_rate):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        with tf.variable_scope(type(self).__name__, reuse=False):
            # TODO: use tensor array
            var_names = ['Wf', 'bf', 'Wi', 'bi', 'Wo', 'bo', 'Wc', 'bc']
            for var in var_names:
                if var.startswith('W'):
                    tf.get_variable(var, shape=(self.hidden_size, self.hidden_size + self.input_size),
                                    dtype=np.float32, initializer=tf.uniform_unit_scaling_initializer())
                elif var.startswith('b'):
                    tf.get_variable(var, shape=self.hidden_size,
                                        dtype=np.float32, initializer=tf.constant_initializer(0.0))
                else:
                    raise Exception(var)
            # output matrix
            Wout = tf.get_variable("Wout", shape=(self.output_size, self.hidden_size),
                                 dtype=np.float32, initializer=tf.uniform_unit_scaling_initializer())
            bout = tf.get_variable("bout", shape=self.output_size, dtype=np.float32,
                                 initializer=tf.constant_initializer(0.0))
        inputs = tf.placeholder(dtype=tf.float32, shape=[self.num_steps, self.input_size])
        targets = tf.placeholder(dtype=tf.float32, shape=[self.num_steps, self.input_size])
        target_weights = tf.placeholder(dtype=tf.float32, shape=[self.num_steps])
        h = tf.constant(np.zeros(self.hidden_size), dtype=tf.float32)
        c = tf.constant(np.zeros(self.hidden_size), dtype=tf.float32)
        loss = 0
        for t in range(0, num_steps):
            #TODO: dig into the use of variable scope
            with tf.variable_scope(type(self).__name__, reuse=True):
                var_list = []
                for var in var_names:
                    var_list.append(tf.get_variable(var))
                Wf, bf, Wi, bi, Wo, bo, Wc, bc = var_list
                Wout = tf.get_variable("Wout")
                bout = tf.get_variable("bout")
            x = inputs[t, :]
            y = targets[t, :]
            y_weights = target_weights[t]
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
            loss += -tf.reduce_sum(tf.log(output) * y) * y_weights
        loss /= tf.reduce_sum(target_weights)
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        self.inputs = inputs
        self.targets = targets
        self.target_weights = target_weights
        self.loss = loss
        self.train_step = train_step
        

if __name__ == '__main__':
    dataset = cPickle.load(open('MNIST_data/mnist_seq.pkl', 'rb'))
    print '{n} samples in the original dataset'.format(n=len(dataset))
    dataset = [x for x, y in dataset]
    vocab_size = 198 # including eos and padding
    eos = 196
    padding = 197
    seq_size = 50
    dataset_ = []
    for x in dataset:
        len_x = len(x)
        if len_x >= seq_size:
            continue
        elif len_x == seq_size - 1:
            x_ = np.hstack((x, np.array([eos], dtype=x.dtype)))
        else:
            x_ = np.hstack((x, np.array([eos], dtype=x.dtype), padding * np.ones(seq_size - len_x - 1, dtype=x.dtype)))
        assert(len(x_) == seq_size)
        dataset_.append((x_, len_x + 1))        
    dataset = dataset_
    print '{n} samples in the dataset'.format(n=len(dataset))

    output_size = input_size = vocab_size
    hidden_size = 100
    learning_rate = 1e-3
    num_epochs = 1
    num_steps = seq_size - 1
    
    dataset_ = []
    for x, len_x in dataset:
        X = np.zeros((len(x), input_size), np.float32)
        X[np.arange(0, X.shape[0]), x] = 1.0
        data = X[0:-1, :]
        target = X[1:, :]
        target_weight = np.zeros(num_steps, dtype=np.float32)
        target_weight[np.arange(len_x - 1)] = 1.0
        dataset_.append((data, target, target_weight))
    dataset = dataset_
    
    np.random.seed(0)
    dataset = [dataset[i] for i in np.random.permutation(len(dataset))]
    split_point = int(0.95 * len(dataset))
    train_set = dataset[0 : split_point]
    valid_set = dataset[split_point :]
    
    print 'building tensor graph...'
    lstm = BasicLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                     num_steps=num_steps, learning_rate=learning_rate)
    print 'tensor graph built.'
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    cnt = 0
    for epoch in range(num_epochs):
        for data, target,  target_weight in train_set:
            _, loss = sess.run([lstm.train_step, lstm.loss],
                               feed_dict={lstm.inputs: data, lstm.targets: target, lstm.target_weights: target_weight})
            cnt += 1
            # validate the model
            if cnt % 400 == 1:
                loss = 0
                for data2, target2,  target_weight2 in valid_set:
                    loss += sess.run(lstm.loss,
                                     feed_dict={lstm.inputs: data2, lstm.targets: target2, lstm.target_weights: target_weight2})
                loss /= len(valid_set)
                print '{n} samples processed, validation loss {l:.4f}'.format(n=cnt, l=loss)
