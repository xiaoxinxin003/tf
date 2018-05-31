# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 12:54:43 2018

@author: focus
"""

import tensorflow as tf

a = tf.constant(2)

b = tf.constant(3)

c = tf.multiply(a, b)

d = tf.add(c, 1)

with tf.Session as sess :
    print(sess.run(d))

