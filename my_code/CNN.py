# -*- coding: utf-8 -*-
import numpy as np
from collections import deque

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
    '''KL距离，相对信息熵 '''
    return (p1 * np.log(p1/p2)) + ((1 - p1) * np.log((1 - p1) / (1 - p2)))

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
    def __init__(self, datas, labels=None, hid_dim=3, train_count=100, rate=0.01, reg=0.01,  sparse=0.01, punish=3,
                 patience_n=None, patience_v=None):
        in_dim = datas.shape[1]
        out_dim = len(set(labels))
        self.X = datas
        self.Y = labels
        self.w1 = np.random.randn(in_dim, hid_dim) #/ np.sqrt(in_dim)   #输入到隐藏层权重
        self.b1 = np.zeros((1, hid_dim))
        self.w2 = np.random.randn(hid_dim, out_dim) #/ np.sqrt(out_dim)   #隐藏到输出层权重
        self.b2 = np.zeros((1, out_dim))
        self.train_count = train_count   #训练次数
        self.rate = rate    #学习速率
        self.reg = reg  #规整化
        self.sparse = sparse  #稀疏参数， 稀疏目标，激活输出的分布概率
        self.punish = punish  #规整化惩罚系数
        self.data_count = len(datas)   #输入数据个数
        self.patience_n = patience_n  #检查连续多少次，cost变化小于patience_v, 用于测试停止训练
        self.patience_v = patience_v
        self.need_check_patience = patience_n and True or False
        self.cost_last = None #
        self.cost_this =None

    def wight_info(self):
        print 'w1:', self.w1
        print 'b1:', self.b1
        print 'w2:', self.w2
        print 'b2:', self.b2

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
        y = self.forward()[0]
        y = np.argmax(y, axis=1)
        Y = self.Y
        right = [y[i] == Y[i] and 1 or 0 for i in range(self.data_count)]
        return 1.0*sum(right) / self.data_count

    def train(self, train_count=500, print_loss=False):
        train_count = train_count or self.train_count
        w1, b1, w2, b2 = self.get_weight()
        X,Y,reg, rate, data_count = self.X, self.Y,self.reg, self.rate, self.data_count
        Yk = trans_dim(Y)

        if self.need_check_patience:
            cost_diff_list = deque([None]*self.patience_n , self.patience_n) #用于记录本次cost和上次cost的差值

        for i in range(0, train_count):
            #print 'I:%s' % i
            #向前传播
            (y, z2, a1, z1) = self.forward()

            #计算损失，检查损失波动，如果连续小于一定的值，则说明接近稳定点，结束训练
            if self.need_check_patience:
                cost = self.cost(y, data_count, Y, reg, w1, w2)
                if i== 0:
                    self.cost_this = cost
                else:
                    self.cost_last = self.cost_this
                    self.cost_this = cost
                    cost_diff_list.append( abs(self.cost_last - self.cost_this) < self.patience_v  )

                if i > self.patience_n:
                    if all(cost_diff_list):
                        print 'patience is ok return by I:%s' % i
                        return

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