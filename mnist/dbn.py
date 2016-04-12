# coding=utf-8
import tensorflow as tf
import numpy as np
import cPickle
import copy
import Image
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from util import tile_raster_images
from tf_util import Struct, GraphWrapper

