# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#创建数据
x = np.linspace(-7, 7, 180)
#激活函数的原始实现
def sigmoid(inputs):
	y = [1/float(1+np.exp(-x) for x in inputs)]
	return y
def relu(inputs):
	y = [x* (x> 0) for x in inputs]
	return y
def tanh(inputs):
	y = [(np.exp(x)-np.exp(-x))/float(np.exp(x)+np.exp(-x)) for x in inputs]
	return y
def softplus(inputs):
	y = [np.log(1+np.exp(x)) for x in inputs]
	return y

#经过tensorflow激活函数处理的各个y值(创建Tensor)
y_sigmoid = tf.nn.sigmoid(x)
y_relu = tf.nn.relu(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)

#创建会话
sess = tf.Session()
#运行
y_sigmoid, y_relu, y_tanh, y_softplus = sess.run([y_sigmoid, y_relu, y_tanh, y_softplus])

#创建各个激活函数的图像
plt.subplot(221)#两行两列  第一个图像
plt.plot(x, y_sigmoid, c="red", label="sigmoid")
plt.ylim(-0.2, 1.2)#y坐标的区间
plt.legend(loc="best")#为了显示label  放在一个最合适的位置上

plt.subplot(222)#两行两列  第一个图像
plt.plot(x, y_relu, c="green", label="relu")
plt.ylim(-1, 6)#y坐标的区间
plt.legend(loc="best")#为了显示label  放在一个最合适的位置上

plt.subplot(223)#两行两列  第一个图像
plt.plot(x, y_tanh, c="blue", label="tanh")
plt.ylim(-1.3, 1.3)#y坐标的区间
plt.legend(loc="best")#为了显示label  放在一个最合适的位置上

plt.subplot(224)#两行两列  第一个图像
plt.plot(x, y_softplus, c="yellow", label="softplus")
plt.ylim(-1, 6)#y坐标的区间
plt.legend(loc="best")#为了显示label  放在一个最合适的位置上

#显示图像
plt.show()
#关闭会话
sess.close()
