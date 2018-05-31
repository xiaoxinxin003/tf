# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#下载并载入MNIST手写数字库
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

#one_hot  独热码的编码形式
#0 1 2 3 4 5 6 7 8 9
#0 ：1000000000
#1 ：0100000000
#2 ：0010000000
#3 ：0001000000
#4 ：0000100000
#5 ：0000010000
#6 ：0000001000
#7 ：0000000100
#8 ：0000000010
#9 ：0000000001

#None表示张量（Tensor）的第一个维度   可以是任何长度
input_x = tf.placeholder(tf.float32, [None, 28*28])/255.
output_y = tf.placeholder(tf.int32, [None, 10]) #输出10个数字的标签
input_x_images = tf.reshape(input_x, [-1, 28, 28,1])

#从Test（测试）数据集里面选取3000个手写数字的图片和对应的标签
test_x = mnist.test.images[1:3000] #图片
test_y = mnist.test.labels[1:3000] #标签

#构建我们的卷积神经网络
#第一层卷积
conv1 = tf.layers.conv2d(
        inputs=input_x_images,  #形状[28*28*1]
        filters=32,   #32个过滤器，输出深度是32
        kernel_size=[5,5], #过滤器在二维的大小是[5*5]
        strides=1, #步长是1
        padding='same', #same表示输出的大小不变，因此需要在外围补零两圈
        activation=tf.nn.relu #激活函数relu
        )#形状[28,28,32]
#第一层池化（亚采样）
pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2,2], #过滤器在二维的大小是  2*2
        strides=2 #步长是2
        )#形状是[14*14*32]
#构建第二层卷积
conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        strides=1, 
        padding='same',
        activation=tf.nn.relu
        )#形状是[14,14,64]
#构建第二层池化（亚采样）
pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2,2], #过滤器在二维的大小是  2*2
        strides=2 #步长是2
        )#形状是[7*7*64]
#平坦化（flat）
flat = tf.reshape(pool2, [-1, 7*7*64]) #形状[7*7*64] -1代表tf根据之后确定的参数推断-1位置上维度的大小。

#1024个神经元的全链接层
dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

#Dropout 丢弃 50%  rate=0.5
dropout = tf.layers.dropout(inputs=dense, rate=0.5)

#10个神经元的全链接层，这里不用激活函数来做非线性了
logits = tf.layers.dense(inputs=dropout, units=10) #输出，形状[1,1,10]

#计算误差， （计算Cross entropy(交叉熵)，再用softmax计算百分比概率）
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)
#使用Adam优化器来最小化误差，学习率：0.001
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
#精度。计算   预测值和  实际标签的匹配程度
#返回（accuracy, update_op）,会创建两个局部变量
accuracy = tf.metrics.accuracy(
        labels=tf.argmax(output_y, axis=1),
        predictions=tf.argmax(logits, axis=1)[1]
        )
#创建会话
sess = tf.Session()
#初始化变量 :全局和局部
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
sess.run(init)

for i in range(20000):
    batch = mnist.train.next_batch(50) #从Train数据集里取到下一个  50 个样本
    train_loss = sess.run([loss,train_op],{input_x:batch[0],output_y:batch[1]})
    if i% 100 == 0:
        test_accuracy = sess.run(accuracy,{input_x:test_x, output_y:test_y})
        print("Step=%d, Train_loss=%.4f, [Test_accuracy=%.2f]" % (i, train_loss, test_accuracy))
#测试  打印20个 预测值和真实值  对
test_output = sess.run(logits, {input_x:test_x[:20]})
inferenced_y = np.argmax(test_output,1)
print(inferenced_y, 'Inferenced Numbers') #推测的数据
print(np.argmax(test_y[:20],1),'Real Numbers') #真实的数据
















