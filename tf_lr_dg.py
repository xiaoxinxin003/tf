# -*- encoding:utf-8 -*-

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt


x_data = np.random.rand(100)
y_data = 0.3*x_data+2.3

plt.plot(x_data, y_data)

w = tf.Variable(0.)
b = tf.Variable(0.)
y = w*x_data+b

loss = tf.reduce_mean(tf.square(y_data-y))

optimizer = tf.train.GradientDescentOptimizer(0.3)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(203):
        sess.run(train)
        if step %10 == 0:
            print(step, sess.run(w), sess.run(b))
plt.show()
