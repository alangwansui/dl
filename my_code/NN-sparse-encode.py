# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import math
import scipy
import time



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
    def __init__(self, datas, hid_dim, train_counts=1, rate=0.005, reg_lambda=0.01, rho=0.01,  penalty=3):
        in_dim = datas.shape[1]
        out_dim = datas.shape[1]

        self.hid_dim = hid_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w1 = np.random.randn(in_dim, hid_dim) #/ np.sqrt(in_dim)   #输入到隐藏层权重
        self.b1 = np.zeros((1, hid_dim))
        self.w2 = np.random.randn(hid_dim, out_dim) #/ np.sqrt(out_dim)   #隐藏到输出层权重
        self.b2 = np.zeros((1, out_dim))
        self.x = datas

        self.train_counts = train_counts   #训练次数
        self.rate = rate                   #学习速率
        self.reg_lambda = reg_lambda       #规整化参数
        self.count_datas = len(datas)      #输入数据个数

        self.rho = rho                     #稀疏限制
        self.penalty = penalty             #惩罚因子
        self.cost_last = None
        self.cost_this = None

    def print_w(self):
        #打印w参数
        print '===W info:', self.w1.shape,  self.w2.shape,
        print self.w1
        print self.b1
        print self.w2
        print self.b2
        print '==='

    def train(self, train_counts=None, log_flag=False):
        def cost():
            return True


        train_counts = train_counts or self.train_counts  #可以指定新的训练次数，也可以使用初始化参数
        for i in range(0, train_counts):
            print '-----',i
            w1, w2, b1 ,b2, X = self.w1,self.w2,self.b1,self.b2, self.x
            reg_lambda,rate, count_datas = self.reg_lambda, self.rate, self.count_datas

            #向前传导 计算隐藏层输出
            z1 = X.dot(w1) + b1
            a1 = tanh(z1)
            z2 = a1.dot(w2) + b2
            out = tanh(z2)

            #print out
            #计算差距 自编码算法，不存在y， 直接和 x 比较
            diff = out - X


            #cost = diff + ref  + punish  = 1/2error**2 +
            cost = 0.5 * np.sum(np.multiply(diff, diff)) / self.count_datas


            print cost



            #逆向求导，根据链式求导规则推到
            dW2 = (a1.T).dot(diff)
            db2 = np.sum(diff, axis=0, keepdims=True)
            delta2 = diff.dot(w2.T) * d_tanh(a1)
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


    def show(self):
        w1 = self.w1
        w2 = self.w2
        b1 = self.b1
        b2 = self.b2
        X = self.x

        z1 = X.dot(w1) + b1


        print z1.shape





def normalizeDataset(dataset):

    """ Remove mean of dataset """

    dataset = dataset - np.mean(dataset)

    """ Truncate to +/-3 standard deviations and scale to -1 to 1 """

    std_dev = 3 * np.std(dataset)
    dataset = np.maximum(np.minimum(dataset, std_dev), -std_dev) / std_dev

    """ Rescale from [-1, 1] to [0.1, 0.9] """

    dataset = (dataset + 1) * 0.4 + 0.1

    return dataset

def loadDataset(num_patches=10, patch_side=8):

    #载入图片，数据初始化
    images = scipy.io.loadmat('D:\workspace\dl\Sparse-Autoencoder\Sparse-Autoencoder-master\IMAGES.mat')
    images = images['IMAGES']
    dataset = np.zeros((patch_side*patch_side, num_patches))

    """ Initialize random numbers for random sampling of images
        There are 10 images of size 512 X 512 """

    rand = np.random.RandomState(int(time.time()))
    image_indices = rand.randint(512 - patch_side, size = (num_patches, 2))
    image_number  = rand.randint(10, size = num_patches)

    """ Sample 'num_patches' random image patches """

    for i in xrange(num_patches):

        """ Initialize indices for patch extraction """

        index1 = image_indices[i, 0]
        index2 = image_indices[i, 1]
        index3 = image_number[i]

        """ Extract patch and store it as a column """

        patch = images[index1:index1+patch_side, index2:index2+patch_side, index3]
        patch = patch.flatten()
        dataset[:, i] = patch

    """ Normalize and return the dataset """

    dataset = normalizeDataset(dataset)
    return dataset





def main():
    datas = loadDataset(100,8)
    nn = NN(datas, None, 4)
    nn.train(5)
    nn.show()



if __name__ == "__main__":
    main()






