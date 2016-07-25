# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
import cPickle
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__':
    dataset = cPickle.load(open('MNIST_data/mnist_seq.pkl', 'rb'))
    print '{n} samples in the original dataset'.format(n=len(dataset))
    dataset = [x for x, y in dataset]
    input_size = vocab_size = 198 # including eos and padding
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
    num_epochs = 10
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
    split_point = int(0.9 * len(dataset))
    train_set = dataset[0 : split_point]
    valid_set = dataset[split_point :]
    print '{nt} training samples, {nv} validation samples'.format(nt=len(train_set), nv=len(valid_set))

    var_list, _ = cPickle.load(open("model/lstm_epoch8.pkl", "rb"))
    Wf, bf, Wi, bi, Wo, bo, Wc, bc, Wout, bout = var_list
    x = tf.placeholder(dtype=tf.float32, shape=input_size)
    h = tf.placeholder(shape=hidden_size, dtype=tf.float32)
    c = tf.placeholder(shape=hidden_size, dtype=tf.float32)
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
    new_c = f * c + i * cand_c
    # new hidden state
    new_h = o * tf.tanh(new_c)
    lin_output = tf.reshape(tf.matmul(Wout, tf.reshape(new_h, [-1, 1])), [-1]) + bout
    output = tf.reshape(tf.nn.softmax(tf.reshape(lin_output, [1, -1])), [-1])

    first_input = valid_set[2][0][0]
    input_data = first_input
    print 'first input {i}'.format(i=np.argmax(input_data))
    output_list = [np.argmax(input_data)]
    i = 0
    with tf.Session() as sess:
        hidden_state = np.zeros(shape=hidden_size, dtype=np.float32)
        cell_state = np.zeros(shape=hidden_size, dtype=np.float32)
        while True:
            i += 1
            [out_proba, hidden_state, cell_state] = sess.run([output, new_h, new_c], feed_dict={x: input_data, h: hidden_state, c: cell_state})
            out = np.argmax(out_proba)
            print '{i}-th output {o}'.format(i=i, o=out)
            if out == eos or out == padding: break
            output_list.append(out)
            #if out < eos:
            #    output_list.append(out)
            input_data = np.zeros(out_proba.shape, dtype=out_proba.dtype)
            input_data[np.argmax(out)] = 1.0
            if i > seq_size * 2:
                print 'warning: the sequence is too long: {i} entries'.format(i=i)
                break
    im = np.zeros(shape=14*14, dtype=np.uint8)
    im[np.array(output_list)] = 255
    im = im.reshape((14, 14))
    im_vis = np.stack((im, im, im), axis=2)
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    ax.imshow(im_vis)
    ax.axis('off')
    fig.show()
