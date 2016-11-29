# -*- coding: utf-8 -*-
import numpy as np
from collections import deque
import math
import matplotlib.pyplot as plt
import matplotlib

#所有激活函数的选择宗旨，要方便求导，减少反向传播计算

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def d_sigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def d_tanh(y):
    return 1 - np.power(y, 2)

def KL(p1, p2):
    '''KL距离，相对信息熵，用于计算目标稀疏参数分布 和 实际激活a1分布的距离 '''
    return (p1 * np.log(p1/p2)) + ((1 - p1) * np.log((1 - p1) / (1 - p2)))

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
    z1 = X.dot(w1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(w2) + b2
    y = sigmoid(z2)   #计算分布概率
    return (y, z2, a1, z1)

class NN(object):
    def __init__(self, datas, labels=None, hid_dim=3, train_count=10000, rate=0.0001, reg=0.0001,  sparse=0.01, punish=3,
                 patience_n=None, patience_v=None):

        in_dim = datas.shape[1]  #输入维度
        out_dim = labels and len(set(labels)) or in_dim  #自编算法，输出维度和输入维度一样，如果是BP算法，输出维度由labels决定
        hid_dim = hid_dim**2  # 隐藏层维度
        self.X = datas
        self.Y = labels
        #r = math.sqrt(6) / math.sqrt(in_dim + out_dim + 1)
        #rand = np.random.RandomState(1)
        #self.w1 = np.asarray(rand.uniform(low = -r, high = r, size = (in_dim, hid_dim)))
        #self.w2 = np.asarray(rand.uniform(low = -r, high = r, size = (hid_dim, out_dim)))

        self.w1 = np.ones((in_dim, hid_dim)) * 0.003 #/ np.sqrt(in_dim)   #输入到隐藏层权重，可以用随机值，采用固定0.001，方便debug
        self.b1 = np.zeros((1, hid_dim))
        self.w2 = np.ones((hid_dim, out_dim)) * 0.003 #/ np.sqrt(out_dim)   #隐藏到输出层权重
        self.b2 = np.zeros((1, out_dim))

        self.train_count = train_count  #训练次数
        self.rate = rate    #学习速率
        self.reg = reg  #规整化参数
        self.sparse = sparse  #稀疏参数， 稀疏目标，激活输出的目标分布概率
        self.punish = punish  #惩罚系数
        self.data_count = len(datas)   #样本数量
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

    def get_weight(self):
        return  self.w1, self.b1, self.w2, self.b2

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
        w1, b1, w2, b2 = self.get_weight()
        X = self.X
        return  forward(X, w1, w2, b1, b2)

    def validation(self):
        '''验证准确率'''
        y = self.forward()[0]
        y = np.argmax(y, axis=1)
        Y = self.Y
        right = [y[i] == Y[i] and 1 or 0 for i in range(self.data_count)]
        return 1.0*sum(right) / self.data_count

    def train(self, train_count=1, print_loss=False):
        train_count = train_count or self.train_count
        w1, b1, w2, b2 = self.get_weight()
        reg, rate, punish, sparse, = self.reg, self.rate, self.punish, self.sparse
        X, data_count = self.X, self.data_count

        for i in range(0, train_count):
            #向前传播
            (y, z2, a1, z1) = self.forward()
            #隐藏层激活分布a1
            sparse_cap = np.sum(a1, axis = 0) / data_count  #sparse_cap
            #差距
            diff = y - X

            #代价-差距
            cost_diff = 0.5 * np.sum(np.multiply(diff, diff)) / data_count
            #代价-L2范数
            cost_w = 0.5 * reg * (np.sum(np.multiply(w1, w1)) + np.sum(np.multiply(w2, w2)))
            #代价-稀疏性
            cost_kl = punish * np.sum(KL(sparse, sparse_cap))  #激活度和a1有关，也就是和w1有关，和w2无关，
            #代价-计总
            cost = cost_diff + cost_w + cost_kl

            print 'i:%s cost:%s' % (i, cost)

            #w2 b2 求导， 根据cost函数，反向求导也分成3部分
            delta2 = diff * d_sigmoid(y)  #delta2保存，方便 dw1的计算
            dw2_diff = a1.transpose().dot(delta2) / data_count
            dw2_w =  self.rate * w2
            dw2_kl = 0  #KL计算不包W，所以此部分为0
            dw2 = dw2_diff + dw2_w + dw2_kl
            db2 = np.sum(delta2.transpose(), axis=1) / data_count

            #w1 b1 求导 根据cost函数，反向求导也分成3部分
            #cost_kl 只和a1有关，delta1 的计算要考虑 cost_kl(a1)
            delta1 = delta2.dot(w2.transpose()) * d_sigmoid(a1)
            dw1_diff =  X.transpose().dot(delta1)
            dw1_w =  self.rate * w1
            delta_kl = np.tile(- sparse / sparse_cap + (1 - sparse) / (1 - sparse_cap), (data_count, 1)).transpose()
            dw1_kl = X.transpose().dot(punish * delta_kl.transpose() * d_sigmoid(a1))
            dw1 = dw1_diff + dw1_w + dw1_kl
            db1 = np.sum(delta1.transpose(), axis=1) / data_count

            #更新参数
            w1 += -rate * dw1
            b1 += -rate * db1
            w2 += -rate * dw2
            b2 += -rate * db2
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