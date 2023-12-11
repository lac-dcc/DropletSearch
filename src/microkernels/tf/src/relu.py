import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import time
from tensorflow.python.framework import graph_util
import sys

flags = tf.compat.v1.flags
#logging = tf.logging
#logging.set_verbosity(tf.logging.ERROR)
C = 4096
repeat_time = 1000

if __name__ == "__main__":
    if len(sys.argv) == 2:
        N = int(sys.argv[1])
    print("N, C, repeat_time:", N, C, repeat_time)
    flags.DEFINE_integer("N", N, "N")
    flags.DEFINE_integer("C", C, "C")
    FLAGS = flags.FLAGS
    #tf.enable_eager_execution()
    #print('is eager mode: ',tf.executing_eagerly())
    a = tf.ones([FLAGS.N, FLAGS.C], tf.float32)
    t = tf.reduce_sum(a).numpy()
    st = time.time()
    for i in range(repeat_time):
        c = tf.nn.relu(a)
    x = tf.reduce_sum(c)
    _ = x.numpy()
    ed = time.time()
    print("{} ms on avg".format((ed-st)*1000.0/repeat_time))
    pass
