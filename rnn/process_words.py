# coding=utf-8
import numpy as np
import cPickle


def word2array(s, char2int):
    return np.array([char2int.get(c) for c in s if char2int.get(c) is not None])


def array2word(a, int2char):
    return ''.join([int2char.get(x) for x in a if int2char.get(x) is not None])


if __name__ == '__main__':
    with open('English_words/google-10000-english-usa.txt', 'rb') as f:
        lines = f.readlines()
    words = [w.strip() for w in lines]
    
    chars = 'abcdefghijklmnopqrstuvwxyz'
    char2int = {}
    int2char = {}
    for c, i in zip(chars, range(len(chars))):
        char2int[c] = i
        int2char[i] = c
    
    input_size = vocab_size = 28 # including eos and padding
    eos = 26
    padding = 27
    seq_size = 11 # incl. eos
    num_steps = seq_size - 1
    
    # pad words to a fixed length
    words_padded = []
    for x in words:
        x = word2array(x, char2int)
        len_x = len(x)
        if len_x >= seq_size:
            continue
        elif len_x == seq_size - 1:
            x_ = np.hstack((x, np.array([eos], dtype=x.dtype)))
        else:
            x_ = np.hstack((x, np.array([eos], dtype=x.dtype), padding * np.ones(seq_size - len_x - 1, dtype=x.dtype)))
        assert(len(x_) == seq_size)
        words_padded.append((x_, len_x + 1))
    
    # encode words into onehot sequence
    dataset = []
    for x, len_x in words_padded:
        X = np.zeros((len(x), input_size), np.float32)
        X[np.arange(0, X.shape[0]), x] = 1.0
        data = X[0:-1, :]
        target = X[1:, :]
        target_weight = np.zeros(num_steps, dtype=np.float32)
        target_weight[np.arange(len_x - 1)] = 1.0
        dataset.append((data, target, target_weight))
    cPickle.dump(dataset, open('English_words/dataset.pkl', 'wb'))