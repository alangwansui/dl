# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
#import math



def generate_data(count, noise):
    np.random.seed(0)
    X, y = datasets.make_moons(count, noise=noise)
    return X, y

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def d_sigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def d_tanh(y):
    return 1 - np.power(y, 2)

def softmax(z):
    '''
    相当于 多维度 结果的 sigmoid 函数， 把输出转换为概率
    If we take an input of [1,2,3,4,1,2,3],
    the softmax of that is [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
    '''
    exp_scores = np.exp(z)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs


class NN(object):
    def __init__(self, datas, labels, hid_dim, train_counts=1, rate=0.01, reg_lambda=0.01 ):
        in_dim = datas.shape[1]
        out_dim = len(set(labels))
        self.w1 = np.random.randn(in_dim, hid_dim) #/ np.sqrt(in_dim)   #输入到隐藏层权重
        self.b1 = np.zeros((1, hid_dim))
        self.w2 = np.random.randn(hid_dim, out_dim) #/ np.sqrt(out_dim)   #隐藏到输出层权重
        self.b2 = np.zeros((1, out_dim))
        self.x = datas
        self.y= labels
        self.YY = self.y_trans_dim(labels)
        self.train_counts = train_counts   #训练次数
        self.rate = rate    #学习速率
        self.reg_lambda = reg_lambda  #规整花
        self.count_datas = len(datas)   #输入数据个数

    def print_w(self):
        '''
        打印权重信息
        '''
        print 'weight info:'
        print self.w1, self.b1
        print self.w2,  self.b2

    def y_trans_dim(self, labels):
        ''' [0,1,1,0] ->   [[1,0],[0,1],[0,1],[,0,],]
            [0,1,2]  ->  [1,0,0],[0,1,0],[0,0,1]
        '''
        yy = np.zeros((len(labels), len(set(labels))))
        for i in range(len(labels)):
            yy[i][labels[i]] = 1
        return yy

    def train(self, train_counts=None, log_flag=False):
        train_counts = train_counts or self.train_counts  #可以指定新的训练次数，也可以使用初始化参数
        for i in range(0, train_counts):
            w1, w2, b1 ,b2, X, y = self.w1,self.w2,self.b1,self.b2, self.x, self.y
            reg_lambda,rate, count_datas = self.reg_lambda, self.rate, self.count_datas

            #向前传导, softmax计算所有输出概率
            z1 = X.dot(w1) + b1
            a1 = tanh(z1)
            z2 = a1.dot(w2) + b2
            probs = softmax(z2)

            #计算差距
            # 预测结果y‘ 和 实际结果 y，差值，由于，输出是二维度，
            # x=[[0.1,0.8],[0.7,0.3]] ,y=[0,1]  x[range(len(x)),y] =  [[0.1, 0.8-1],[0.7-1,0.3]]
            #去 x 对应的y下标为内容0.8，是对应实际维度的预测值，减去y，得到差值
            # 实际上可以把y[1,0,0,1]理解为他的二维表现形式为[[0,1],[1,0],[1,0],[0,1]]，这样和y’ -y 就好理解了

            delta3 = probs
            delta3 -= self.YY   # 因为YY中的非1数据，都是0， 不影响结果，所以 -YY 可以直接计算。
            #delta3[range(self.count_datas), y] -= 1  #此处使用YY多维来直接计算，思路更清晰


            #逆向求导，根据链式求导规则推导
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(w2.T) * d_tanh(a1)
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            # 加上重整话参数, 重整化参数以怎样的方式，影响参数的调整？？
            #dW2 += reg_lambda * w2
            #dW1 += reg_lambda * w1

            # Gradient descent parameter update
            #根据学习速率调整新参数？学习速率实际上可以根据
            w1 += -rate * dW1
            b1 += -rate * db1
            w2 += -rate * dW2
            b2 += -rate * db2

            # Assign new parameters to the model
            #更新参数
            (self.w1, self.b1, self.w2, self.b2) = ( w1,b1, w2, b2)

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.

            #计算损失函数


        return True



    def test(self):
        w1 = self.w1
        w2 = self.w2
        b1 = self.b1
        b2 = self.b2
        X = self.x
        y = self.y
        z1 = X.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        probs = softmax(z2)
        res = np.argmax(probs, axis=1)
        judge = [res[i] == y[i] and 1 or 0  for i in range(self.count_datas)]
        percent = 100.0 * sum(judge) / self.count_datas
        print 'test result:%s' % percent

    def visual(self):
        plt.scatter(self.x[:,0], self.x[:,1], s=20, c=self.y, cmap=plt.cm.Spectral)
        plt.show()


    def visual_w(self, N=100, max=2,min=-2):
        #构建 取值范围为min~max的整屏数据，根据w计算分类，然后显图像，描述权重和截距（W，b）的图像边界
        #方便理解w的几何意义
        step = 1.0*(max-min)/N
        Grids = np.zeros((N**2,2))
        for j in range(-N/2,N/2):
            v1 = j * step
            for k in range(-N/2, N/2):
                v2 = k * step
                index = j*N +k
                Grids[index][0] = v1
                Grids[index][1] = v2
        z1 = Grids.dot(self.w1) + self.b1
        a1 = tanh(z1)
        z2 = a1.dot(self.w2) + self.b2
        probs = softmax(z2)
        Y = np.argmax(probs, axis=1)
        plt.scatter(Grids[:,0],Grids[:,1], s=8, c=Y, cmap=plt.cm.Spectral)
        plt.show()


def main():
    datas,labels = generate_data(200,0.2)
    nn = NN(datas,labels, 3)
    nn.train(500)
    nn.test()
    #nn.visual()
    nn.visual_w(100)




if __name__ == "__main__":
    main()






