# -*- coding: utf-8 -*-
import numpy as np
from collections import deque
import logging


#所有激活函数的选择宗旨，要方便求导，减少反向传播计算
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def d_sigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def d_tanh(y):
    return 1 - np.power(y, 2)

def trans_dim(labels):
    '''
    Y--> Yk, Yk 是只有正确分类为1，其余项为0的向量，
    [0,1,2] -->  [0,0,1],[0,1,0],[0,0,1]
    '''
    yy = np.zeros((len(labels), len(set(labels))))
    for i in range(len(labels)):
        yy[i][labels[i]] = 1
    return yy

def soft_max(z):
    '''
    #soffmax的求导是最重要的一步  http://www.cnblogs.com/ZJUT-jiangnan/p/5791115.html
    相当于 多维度结果的 sigmoid 函数， 把输出转换为概率, EXP(),扩大数据的百分比差距， 构造可以让求导更加简单
    [1，1，2], [0.25,0.25,0.5]
    '''
    exps = np.exp(z)  #为什么要使用 exp() ，就是为了让后面的Loss计算方便，让 delate = y- Yk
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    return probs

def forward(X, w1, w2, b1, b2):
    z1 = X.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    y = soft_max(z2)   #计算分布概率
    return (y, z2, a1, z1)

class NN(object):
    def __init__(self, X, Y, hid_dim=3, train_count=100, rate=0.01, dec=0.01):
        in_dim = X.shape[1]
        out_dim = len(set(Y))
        self.X = X
        self.Y = Y
        self.w1 = np.random.randn(in_dim, hid_dim) #/ np.sqrt(in_dim)   #输入到隐藏层权重
        self.b1 = np.zeros((1, hid_dim))
        self.w2 = np.random.randn(hid_dim, out_dim) #/ np.sqrt(out_dim)   #隐藏到输出层权重
        self.b2 = np.zeros((1, out_dim))
        self.train_count = train_count   #训练次数
        self.rate = rate    #学习速率



    def cost(self, y, data_count, Y, reg, w1, w2):
        """
        #损失函数为什么选  -np.log(y[range(data_count), Y])，也是为了求导方便
        # error = -np.log(out[range(data_count), Y])
        # loss = error_y  +  error_reg, err_y = -np.log(y[range(data_count), Y])
        """
        error_y = -np.log(y[range(data_count), Y])
        error_y = np.sum(error_y)
        error_reg = reg / 2.0 * (  np.sum(np.square(w1)) + np.sum(np.square(w2))  )
        cost = (error_y + error_reg) / data_count
        return cost

    def forward(self):
        X, w1, b1, w2, b2 = self.X, self.w1, self.b1, self.w2, self.b2
        z1 = X.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        y = soft_max(z2)  # 计算分布概率
        return (y, z2, a1, z1)

    def train(self, print_loss=False):
        train_count = self.train_count
        w1, b1, w2, b2 = self.X, self.w1, self.b1, self.w2, self.b2
        X, Y, rate, data_count = self.X, self.Y, self.rate, self.data_count
        Yk = trans_dim(Y)

        for i in range(train_count):
            (y, z2, a1, z1) = self.forward()

            #反向传播 cost 是2项目之和（error_y  error_reg）， 偏导等于2项目偏导之和
            #soffmax的求导 http://www.cnblogs.com/ZJUT-jiangnan/p/5791115.html
            # d(L/w2) = d(L/z2) * d(z2/w2),因为损失函数使用是soft_max, 所以d(L/z2) = y-Yk  其中YK是正确分类为1，其他分类为0的向量
            #d(L/z2) 对下层求导也有用， 提取记为  delta2
            #所以  dw2 = delta2 * d(z2/w2)
            delta2 = y - Yk
            dw2 = (a1.T).dot(delta2)
            db2 = np.sum(delta2, axis=0, keepdims=True)

            #dw1 = d(L/z2)  *  d(z2/a1)  *  d(a1/z1) * d(a1/w1)
            #dw1 = delta2   *    W2         1-a1^2  *    X
            delta1 = delta2.dot(w2.T) * d_tanh(a1)
            dw1 = np.dot(X.T, delta1)
            db1 = np.sum(delta1, axis=0)

            #求规整项目error_reg的导数 d(ref/2*w^2) = ref*w
            dw2 += self.reg * w2
            dw1 += self.reg * w1

            # Gradient descent parameter update
            w1 += -rate * dw1
            b1 += -rate * db1
            w2 += -rate * dw2
            b2 += -rate * db2
            (self.w1, self.b1, self.w2, self.b2) = (w1,b1, w2, b2)

        return True


    def validation(self):
        #验证结果
        y = self.forward()[0]
        y = np.argmax(y, axis=1)
        Y = self.Y
        right = [y[i] == Y[i] and 1 or 0 for i in range(self.data_count)]
        return 1.0*sum(right) / self.data_count

    def wight_info(self):
        print 'w1:', self.w1
        print 'b1:', self.b1
        print 'w2:', self.w2
        print 'b2:', self.b2