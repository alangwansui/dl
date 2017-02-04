#coding:utf-8
import numpy as np
import numpy as np
from collections import deque
import math
import matplotlib.pyplot as plt
import matplotlib



# -*- coding: utf-8 -*-
from TNN import *
from data_load import generate_data_minist
import matplotlib.pyplot as plt


def show(datas, labels, w1,b1,w2,b2, N=100, max=3,min=-3,):
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

    y_Grids = np.argmax(forward(Grids, w1, w2, b1, b2)[0], axis=1)
    plt.scatter(Grids[:,0],Grids[:,1], s=6, c=y_Grids, cmap=plt.cm.Spectral) # 权重信息可视化

    plt.scatter(datas[:,0],datas[:,1], s=12, c=labels, cmap=plt.cm.Spectral) # 实际数据可视化

    plt.show()

#################################################################
datas, labels= generate_data_minist()
nn = NN(datas, labels, 50, patience_n=6, patience_v=0.001)
nn.train(5)
print 'right rate: %s' % nn.validation()


print '__END OK__'


