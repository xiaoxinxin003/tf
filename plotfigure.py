# -*- coding: utf-8 -*-
"""
Created on Wed May  9 22:06:25 2018
练习使用matplotlib
@author: focus
"""

#引入matplotlib子包pyplot
import matplotlib.pyplot as plt
import numpy as np

#创建数据
x = np.linspace(-4, 4, 60)
#y = 3 * x + 4
y1 = 3 * x + 4
y2 = x ** 2

#创建第一个图像
plt.figure(num = 1, figsize=(5, 5))
plt.plot(x, y1)
plt.plot(x, y2, color="red", linewidth=2.0, linestyle="--")
#创建第二个图
plt.figure(num=2)
plt.plot(x, y2)
#显示图像
plt.show()


