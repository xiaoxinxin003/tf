# -*- coding:utf-8 -*-

import tensoflow as tf

#创建一个先入先出队列，初始化队列插入0.1 0.2  0.3数字
q = tf.FIFOQueue(3, "float")
init = q.enqueue_many(([0.1,0.2,0.3],))
#定义出队、+1、入队操作
x = q.dequeue()
y = x+1
q_inc = q.enqueue([y])
