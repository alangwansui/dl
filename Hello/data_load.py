#coding:utf-8
import gzip
import cPickle as pickle
import numpy as np
from sklearn import datasets, linear_model

#手写数字识别数据
def generate_data_minist(count=None):
    filename = 'D:/workspace/dl/data/mnist.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)
    X_train, y_train = data[0]

    return X_train, y_train


    X_val, y_val = data[1]
    X_test, y_test = data[2]
    X_train = X_train.reshape((-1, 1, 28, 28))
    X_val = X_val.reshape((-1, 1, 28, 28))
    X_test = X_test.reshape((-1, 1, 28, 28))
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    return X_train, y_train, X_val, y_val, X_test, y_test


##创建二维分类数据
def generate_data(count, noise):
    np.random.seed(0)
    X, y = datasets.make_moons(count, noise=noise)
    return X, y


if __name__ == '__main__':
    X = generate_data(100,1)[0]
    print X.shape


