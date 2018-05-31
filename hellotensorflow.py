# -*- coding: utf-8 -*-
import tensorflow as tf

hello = tf.constant("hello tensorflow ")

sess = tf.Session()

print(sess.run(hello))

sess.close