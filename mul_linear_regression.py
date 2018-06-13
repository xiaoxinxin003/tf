#-*-encoding:utf-8-*-

from numpy import genfromtxt
import numpy as np
from sklearn import datasets,linear_model

data_path = r"e:\code\tf\data\test.csv"

delivary_data = genfromtxt(data_path,delimiter=',')

print("data")
print(delivary_data)
#第一个冒号代表所有行，第二个冒号：从0列到-1列,但是不包括倒数第一列。
X = delivary_data[:,:-1]
#所有行  只要第-1列
Y = delivary_data[:,-1]

print("X")
print(X)
print("Y")
print(Y)

regr = linear_model.LinearRegression()
#model.fit 表示对数据进行数学建模
regr.fit(X,Y)
#打印 b0  b1  b2
print("coefficients")
print(regr.coef_)
print("intercept")
print(regr.intercept_)

#预测
xpred = [102 , 6]
ypred = regr.predict(xpred)
print("predy is;")
print(ypred)
