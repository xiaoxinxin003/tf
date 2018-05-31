
"""
tensorflow 加载数据的方式有三种：
1.预加载数据，在tensorflow图中定义变量或者常量来保存所有数据
2.填充数据,python产生数据，然后填充后端
3.从文件读取数据，让队列管理器从文件中读取数据。
队列管理器见：tf_queue_***.py
"""
import tensorflow as tf

p1 = tf.placeholder(tf.int16)
p2 = tf.placeholder(tf.int16)

a1 = tf.add(p1, p2)

#使用Python产生数据然后喂给tensorflow
li1 = [2,3,4]
li2 = [3,5,9]

with tf.Session() as sess:
	print(sess.run(a1, feed_dict={p1:li1, p2:li2}))
