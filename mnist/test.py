# coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops

#x = tf.constant(np.array([1, 2, 3]))
#y = tf.Variable(0)
#for i in tf.range(0, x.get_shape()[0]):
#    y.assign_add(x[i])
#sess = tf.Session()
#sess.run(tf.initialize_all_variables())
#print sess.run(y)
#sess.close()

#i = tf.constant(0)
#c = lambda x: math_ops.less(x, 10)
#b = lambda x: math_ops.add(x, 1)
#r = control_flow_ops.While(c, b, [i])
#with tf.Session() as sess:
#    print sess.run(r)
