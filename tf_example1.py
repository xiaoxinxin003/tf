# -*- coding: utf-8 -*-
import tensorflow as tf

matrix1 = tf.constant([[3,6]])

matrix2 = tf.constant([[4],[8]])

product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    result = sess.run([product])
    print(result)
