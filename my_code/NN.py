# -*- coding: utf-8 -*-
import numpy as np
from collections import deque
import math
import matplotlib.pyplot as plt
import matplotlib

#所有激活函数的选择宗旨，要方便求导，减少反向传播计算

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def d_tanh(y):
    return 1 - np.power(y, 2)

def KL(x, y):
    '''KL距离，相对信息熵，用于计算目标稀疏参数分布 和 实际激活a1分布的距离 '''
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

def d_KL(p1, p2, count):
    np.tile(- p1 / p2 + (1 - p1) / (1 - p2), (count, 1)).transpose()

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
    print 'forwa', X.shape, w1.shape,  X.dot(w1).shape

    z1 = X.dot(w1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(w2) + b2
    a2 = sigmoid(z2)   #计算分布概率
    return (a2, z2, a1, z1)

class Encoder(object):
    def __init__(self, datas, vis_patch_side=8, hid_patch_side=5, train_count=10000, rate=0.0001, reg=0.0001,  sparse=0.01, punish=3,
                 patience_n=None, patience_v=None):

        input_dim = out_dim = vis_patch_side ** 2  #输入维度
        hid_dim = hid_patch_side ** 2  # 隐藏层维度
        self.X = datas
        self.hid_dim = hid_dim
        self.input_dim = input_dim
        #r = math.sqrt(6) / math.sqrt(in_dim + out_dim + 1)
        #rand = np.random.RandomState(1)
        #self.w1 = np.asarray(rand.uniform(low = -r, high = r, size = (in_dim, hid_dim)))
        #self.w2 = np.asarray(rand.uniform(low = -r, high = r, size = (hid_dim, out_dim)))

        self.w1 = np.ones((hid_dim, input_dim),) * 0.001 #/ np.sqrt(in_dim)   #输入到隐藏层权重，可以用随机值，采用固定0.001，方便debug
        self.b1 = np.zeros(hid_dim, dtype=np.float64)
        self.w2 = np.ones((input_dim, hid_dim), ) * 0.001 #/ np.sqrt(out_dim)   #隐藏到输出层权重
        self.b2 = np.zeros(out_dim, dtype=np.float64)

        self.train_count = train_count  #训练次数
        self.rate = rate    #学习速率
        self.reg = reg  #规整化参数
        self.sparse = sparse  #稀疏参数， 稀疏目标，激活输出的目标分布概率
        self.punish = punish  #惩罚系数
        self.data_count = datas.shape[1]   #样本数量
        self.patience_n = patience_n  #检查耐心。连续多少次，cost变化小于patience_v, 用于达到目标后提前退出训练
        self.patience_v = patience_v  #检查耐心，。。。
        self.need_check_patience = patience_n and True or False  #检查耐心，。。。
        self.cost_last = None #检查耐心，。。。
        self.cost_this =None  #检查耐心，。。。

    def print_weight_shape(self):
        print 'w1:', self.w1.shape
        print 'b1:', self.b1.shape
        print 'w2:', self.w2.shape
        print 'b2:', self.b2.shape

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
        W1,W2,b1,b2 = self.w1, self.w2, self.b1, self.b2
        data, count = self.X, self.data_count
        #print W1.dot(data).shape, np.tile(b1, (count, 1)).shape
        #print  np.zeros((self.hid_dim,1)).shape###
        z2 = W1.dot(data) + np.tile(b1, (count, 1)).transpose()
        a2 = sigmoid(z2)
        z3 = W2.dot(a2) + np.tile(b2, (count, 1)).transpose()
        h = sigmoid(z3)
        #print 'h', h.shape, 'x', data.shape, W1.shape, W2.shape
        return (h, z3, a2, z2)

    def train(self, train_count=1, print_loss=False):
        train_count = train_count or self.train_count
        w1, b1, w2, b2 = self.w1, self.b1, self.w2, self.b2
        reg, rate, punish, sparse = self.reg, self.rate, self.punish, self.sparse
        X, sample_count = self.X, self.data_count
        hid_dim = self.hid_dim

        for i in range(0, train_count):
            print i
            #向前传播
            (h, z3, a2, z2) = self.forward()

            # Sparsity
            rho_hat = np.sum(a2, axis=1) / sample_count
            rho = np.tile(sparse, hid_dim)

            # Cost function
            cost_diff = np.sum((h - X) ** 2) / (2 * sample_count)
            cost_w = (rate / 2) * (np.sum(w1 ** 2) + np.sum(w2 ** 2))
            cost_kl =  punish * np.sum(KL(rho, rho_hat))
            cost =  cost_diff + cost_w + cost_kl

            #print 'cost', cost

            # Backprop
            sparsity_delta = np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (sample_count, 1)).transpose()

            delta3 = -(X - h) * d_sigmoid(z3)
            delta2 = (w2.transpose().dot(delta3) + punish * sparsity_delta) * d_sigmoid(z2)
            W1grad = delta2.dot(X.transpose()) / sample_count + rate * w1
            W2grad = delta3.dot(a2.transpose()) / sample_count + rate * w2
            b1grad = np.sum(delta2, axis=1) / sample_count
            b2grad = np.sum(delta3, axis=1) / sample_count

            #更新参数,? 更新参数需要 * rate吗？
            w1 += -rate * W1grad
            b1 += -rate * b1grad
            w2 += -rate * W2grad
            b2 += -rate * b2grad
            (self.w1, self.b1, self.w2, self.b2) = (w1,b1, w2, b2)


    def show(self, n=9):
        #学习结果显示，向前计算，排序，得出最大的N个激活值，获取原始输入图像，即可知道那些xi对W的激励最大，也就是W要寻找的构成
        #如何显示多余10张的子图？，，意义不大，留给你们了
        y, z2, a1, z1 = self.forward()
        X = self.X

        active_values = np.sum(a1, axis=1)
        active_index = active_values.argsort()
        show_index = active_index[-n:]

        for i in range(len(show_index)):
            ss = '55%s' % (i + 1)
            ss = float(ss)
            ss = int(ss)
            subplt = plt.subplot(ss)
            index = show_index[i]
            data = X[index,:]
            subplt.imshow(data.reshape(8, 8), cmap=matplotlib.pyplot.cm.gray, interpolation='nearest')
            subplt.set_frame_on(False)
            subplt.set_axis_off()
        plt.show()
        return True