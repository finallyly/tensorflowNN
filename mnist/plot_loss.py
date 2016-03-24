# coding=utf-8
import cPickle
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.close('all')
    with open('data/validation_loss_1e-2.pkl', 'rb') as f:
        meta2 = cPickle.load(f)
    with open('data/validation_loss_1e-3.pkl', 'rb') as f:
        meta3 = cPickle.load(f)
    with open('data/validation_loss_1e-4.pkl', 'rb') as f:
        meta4 = cPickle.load(f)
    steps = [x[0] for x in meta2]
    loss2 = [x[1] for x in meta2]
    loss3 = [x[1] for x in meta3]
    loss4 = [x[1] for x in meta4]
    
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    ax.plot(steps, loss2, label='1e-2', linewidth=2)
    ax.hold(True)
    ax.plot(steps, loss3, label='1e-3', linewidth=2)
    ax.plot(steps, loss4, label='1e-4', linewidth=2)
    ax.hold(False)
    ax.grid(True)
    ax.set_xlabel('step')
    ax.set_ylabel('validation loss')
    ax.set_title('learning rates')
    ax.legend()
    fig.show()