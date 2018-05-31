# -*- coding: utf-8 -*-
"""
Created on Wed May  9 22:06:25 2018

@author: focus
"""

#引入matplotlib子包pyplot
import matplotlib.pyplot as plt
import numpy as np

#创建数据
x = np.linspace(-2, 2, 100)
#y = 3 * x + 4
y1 = 3 * x + 4
y2 = x ** 2

#创建图像
plt.plot(x, y1)
plt.plot(x, y2)
#显示图像
plt.show()


