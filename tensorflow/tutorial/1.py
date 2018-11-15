import numpy as np
import tensorflow as tf

x = tf.constant([1., 2.])
neg_op = tf.negative(x)

with tf.Session() as sess:
    result = sess.run(neg_op)
print(result)