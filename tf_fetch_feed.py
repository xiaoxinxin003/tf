# -*- encoding:utf-8 -*-
import tensorflow as tf

#FETCH sess中同时运行两个op
a1 = tf.constant(2.0)
a2 = tf.constant(3.0)
a3 = tf.constant(5.0)

add = tf.add(a1, a3)
multi = tf.multiply(a2, a3)

with tf.Session() as sess:
    result = sess.run([multi, add])
    print(result)
#FEED
p1 = tf.placeholder(tf.float32)
p2 = tf.placeholder(tf.float32)
p3 = tf.placeholder(tf.float32)

out = tf.multiply(p1, p2)

with tf.Session() as sess:
    #以字典的形式传入数据
    print(sess.run(out,feed_dict = {p1:[3,],p2:[5.]}))
