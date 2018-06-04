	声明：所有代码位于:~/code/tf/study/下面

认识tf的layers：创建卷及神经网络

tf的layers model提供了高级API去构建一个神经网络，它提供了方法：促进密集（全连接
）层和卷积层的创建，增加激活功能以及应用采样回归，在本教程中，你可以学习到如何使用layers去创建一个卷及神经网络模型，并且使用该模型识别MNIST数据集中的手写数字。

开始：
首先我们来搭建程序的架构，创建文件cnn_mnist.py,增加如下代码：
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

if __name__ == "__main__":
  tf.app.run()


对卷积神经网络的介绍：
卷积神经网络模型是目前图像分类任务中最先进的架构模型了，简称CNNs，CNNs对图片的原始像素数据应用一系列的过滤，抽取并学习高级特征并形成一个模型，日后可以使用该模型对图片进行分类操作，CNNs包含三个模块：
	卷积层：
