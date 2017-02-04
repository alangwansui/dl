#coding:utf-8
import gzip
import cPickle as pickle
import numpy as np
from sklearn import datasets, linear_model

def generate_data_minist(count=None):
    filename = 'D:/workspace/dl/data/mnist.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)
    X, Y = data[0]
    return X,Y

####
x, y  = generate_data_minist()
print x.shape

