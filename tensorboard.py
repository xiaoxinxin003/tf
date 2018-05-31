# -*- coding: utf-8 -*-
"""
Created on Wed May  9 08:41:20 2018

@author: focus
"""

import tensorflow as tf

#构造图（Graph）
#使用一个线性方程为例 y = Wx+b
W = tf.Variable(2.0, dtype=tf.float32, name="Weight")#权重

b = tf.Variable(1.0, dtype=tf.float32, name="Bias")#偏差

x = tf.placeholder(dtype=tf.float32, name="Input")#输入

with tf.name_scope("Output"):
    y = W * x + b  #输出

#定义日志路径
path = "d:/tensorflow/log"

#创建一个操作，用于初始化所有变量。
init = tf.global_variables_initializer()

#创建Session
with tf.Session() as sess:
    sess.run(init) #初始化变量
    writer = tf.summary.FileWriter(path, sess.graph)
    result = sess.run(y, {x:3.0})
    print(" y = %s" % result) #打印W * x + b的值

"""
cd 到log所在目录

tensorboard --logdir=dir/
"""
