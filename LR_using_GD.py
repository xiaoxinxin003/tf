# -*- coding: utf-8 -*-
"""
Created on Thu May 10 13:10:38 2018
使用梯度下降解决线性回归问题
@author: focus
"""

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

#创建数据
point_num = 100
vectors = []
#使用numpy的随机正态分布函数生成100个点
#这些点的x y值对应线性方程：y = 0.1 * x + 0.2
#权重（Wweights）是0.1，偏差（Bias）是0.2
for i in range(point_num):
    x1 = np.random.normal(0.0, 0.66)
    y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
    vectors.append([x1, y1])
#print(vectors)
x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]
#创建图像
plt.plot(x_data, y_data, 'r*', label="Original Data")
plt.title("Linear Regression using Gradient Decent")
plt.legend(loc="best")
plt.show()

#构建线性回归模型
"""
分别创建实际测试中用到的Weight和Bias
"""
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))#初始化Weight
b = tf.Variable(tf.zeros([1])) #初始化 Bias  先初始化一个为0的数  shape 为 1
y = W * x_data + b  #模型计算出来的y值，与y_data是有差别的

#创建log存放的path
path = "d:/tensorflow/log"

#定义损失函数  loss function  也叫 cost function  说明计算出来的拟合曲线和真实点之间的距离偏差 一般用平方衡量
#对Tensor的所有维度计算 ((y-y_dadta) ^ 2)之和， 然后 / N 这里N = point_num = 100
loss = tf.reduce_mean(tf.square(y - y_data)) #reduce_mean 给定一个Tensor，计算该Tensor各个维度的平均值。
#使用梯度下降优化器来优化损失函数 （loss function)
optimizer = tf.train.GradientDescentOptimizer(0.5)#设置学习效率  即步长  一般设置小于1的值
train = optimizer.minimize(loss) #梯度下降最小化损失函数  这样在每次训练时候就会更改W和b值

#创建会话来运行构建的数据流图
sess = tf.Session()

#初始化数据流图中所有的变量
init = tf.global_variables_initializer()
sess.run(init)
#训练20步
for step in range(20):
    #优化每一步
    sess.run(train)
    #writer = tf.summary.FileWriter(path, sess.graph)
    #打印每一步的loss  w 和 b
    print("step=%d, loss=%f, [Weight=%f Bias=%f]" % (step, sess.run(loss), sess.run(W), sess.run(b)))
#创建图

plt.plot(x_data, y_data, 'r*', label="Original Data")
plt.title("Linear Regression using Gradient Decent")
plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label="Fitted Line")
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#关闭会话
sess.close()
