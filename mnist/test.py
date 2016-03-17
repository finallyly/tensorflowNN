# coding=utf-8
import numpy as np
import tensorflow as tf

class GraphWrapper(object):
    def __init__(self, g, var, tsr, ops):
        self.g = g
        self.var = var
        self.tsr = tsr
        self.ops = ops
        
def build_graph(var_val=None):
    g = tf.Graph()
    var = {}
    tsr = {}
    ops = {}
    with g.as_default():
        if var_val is None:
            var['x'] = tf.Variable(1)
            var['y'] = tf.Variable(2)
        else:
            var['x'] = tf.Variable(var_val['x'])
            var['y'] = tf.Variable(var_val['y'])
        tsr['z'] = var['x'] + var['y']
        ops['init_vars'] = tf.initialize_all_variables()
    return GraphWrapper(g, var, tsr, ops)

var_val = {'y': 20, 'x': 10}
gw = build_graph(var_val)
sess = tf.Session(graph=gw.g)
with sess.as_default():
    gw.ops['init_vars'].run()
    print gw.tsr['z'].eval()
    var_val = {}
    for k, v in gw.var.iteritems():
        var_val[k] = v.eval()
    print var_val
    
#g, var, tsr, ops = build_graph()
#sess = tf.Session(graph=g)
#with sess.as_default():
#    ops['init_vars'].run()
#    print tsr['z'].eval()
#    var_val = {}
#    for k, v in var.iteritems():
#        var_val[k] = v.eval()
#    print var_val