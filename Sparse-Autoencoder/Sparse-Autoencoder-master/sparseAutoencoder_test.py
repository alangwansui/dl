# -*- coding: utf-8 -*-
# This piece of software is bound by The MIT License (MIT)
# Copyright (c) 2013 Siddharth Agrawal
# Code written by : Siddharth Agrawal
# Email ID : siddharth.950@gmail.com

import numpy
import math
import time
import scipy.io
import scipy.optimize
import matplotlib.pyplot

###########################################################################################
""" The Sparse Autoencoder class """

class SparseAutoencoder(object):
    def __init__(self, visible_size, hidden_size, rho, rate, penalty):
        self.visible_size = visible_size    # 输入维度
        self.hidden_size = hidden_size      # 隐藏层维度
        self.rho = rho                      # 平均激活度
        self.rate = rate                    # 学习速度
        self.penalty = penalty              # 惩罚参数

        # W 随机初始化  W1.shape = 25*64   W2.shape=64*25
        r = math.sqrt(6) / math.sqrt(visible_size + hidden_size + 1)
        rand = numpy.random.RandomState(int(time.time()))
        W1 = numpy.asarray(rand.uniform(low = -r, high = r, size = (hidden_size, visible_size)))
        W2 = numpy.asarray(rand.uniform(low = -r, high = r, size = (visible_size, hidden_size)))
        print '=====w1:%s------W2:%s======' % (W1.shape, W2.shape)
        b1 = numpy.zeros((hidden_size, 1))
        b2 = numpy.zeros((visible_size, 1))

        #为后面的reshape 做准备
        self.limit0 = 0
        self.limit1 = hidden_size * visible_size
        self.limit2 = 2 * hidden_size * visible_size
        self.limit3 = 2 * hidden_size * visible_size + hidden_size
        self.limit4 = 2 * hidden_size * visible_size + hidden_size + visible_size
        #参数扁平化，
        self.theta = numpy.concatenate((W1.flatten(), W2.flatten(),
                                        b1.flatten(), b2.flatten()))


    #sigmoid函数
    def sigmoid(self, x):
        return (1 / (1 + numpy.exp(-x)))

    #计算残差的
    def sparseAutoencoderCost(self, theta, input):
        #参数 reshape
        W1 = theta[self.limit0 : self.limit1].reshape(self.hidden_size, self.visible_size)
        W2 = theta[self.limit1 : self.limit2].reshape(self.visible_size, self.hidden_size)
        b1 = theta[self.limit2 : self.limit3].reshape(self.hidden_size, 1)
        b2 = theta[self.limit3 : self.limit4].reshape(self.visible_size, 1)

        #向前传播
        hidden_layer = self.sigmoid(numpy.dot(W1, input) + b1)
        output_layer = self.sigmoid(numpy.dot(W2, hidden_layer) + b2)

        #估计隐藏层的平均激活值, 影藏层之和/样本个数.
        rho_cap = numpy.sum(hidden_layer, axis = 1) / input.shape[1]

        #计算距离
        diff = output_layer - input

        #计算距离的平均方差  1/2(diff**2)   为什么用这个方差  因为 1/2(X**2) 的导数刚好等于x，后面方向求导方便
        sum_of_squares_error = 0.5 * numpy.sum(numpy.multiply(diff, diff)) / input.shape[1]

        #权重权重衰减
        weight_decay = 0.5 * self.rate * (numpy.sum(numpy.multiply(W1, W1)) +
                                                   numpy.sum(numpy.multiply(W2, W2)))

        #相对煽距离
        KL_divergence = self.penalty * numpy.sum(self.rho * numpy.log(self.rho / rho_cap) +
                                                    (1 - self.rho) * numpy.log((1 - self.rho) / (1 - rho_cap)))


        #损失等于 目标均方差 + 相对煽距离 + 权重衰减
        cost = sum_of_squares_error + weight_decay + KL_divergence


        KL_div_grad = self.penalty * (-(self.rho / rho_cap) + ((1 - self.rho) / (1 - rho_cap)))

        #out层的的导数 * 差距。等于W1的差距
        del_out = numpy.multiply(diff, numpy.multiply(output_layer, 1 - output_layer))

        #hid层的导数 *  W2的差距
        del_hid = numpy.multiply(numpy.dot(numpy.transpose(W2), del_out) + numpy.transpose(numpy.matrix(KL_div_grad)), 
                                 numpy.multiply(hidden_layer, 1 - hidden_layer))
        
        # 差距
        W1_grad = numpy.dot(del_hid, numpy.transpose(input))
        W2_grad = numpy.dot(del_out, numpy.transpose(hidden_layer))
        b1_grad = numpy.sum(del_hid, axis = 1)
        b2_grad = numpy.sum(del_out, axis = 1)
            
        W1_grad = W1_grad / input.shape[1] + self.rate * W1
        W2_grad = W2_grad / input.shape[1] + self.rate * W2
        b1_grad = b1_grad / input.shape[1]
        b2_grad = b2_grad / input.shape[1]
        

        #更新W参数
        W1_grad = numpy.array(W1_grad)
        W2_grad = numpy.array(W2_grad)
        b1_grad = numpy.array(b1_grad)
        b2_grad = numpy.array(b2_grad)
        theta_grad = numpy.concatenate((W1_grad.flatten(), W2_grad.flatten(),
                                        b1_grad.flatten(), b2_grad.flatten()))
        print '--', cost, theta_grad
        return [cost, theta_grad]

###########################################################################################
""" Normalize the dataset provided as input """

def normalizeDataset(dataset):

    """ Remove mean of dataset """

    dataset = dataset - numpy.mean(dataset)
    
    """ Truncate to +/-3 standard deviations and scale to -1 to 1 """
    
    std_dev = 3 * numpy.std(dataset)
    dataset = numpy.maximum(numpy.minimum(dataset, std_dev), -std_dev) / std_dev
    
    """ Rescale from [-1, 1] to [0.1, 0.9] """
    
    dataset = (dataset + 1) * 0.4 + 0.1
    
    return dataset

###########################################################################################
""" Randomly samples image patches, normalizes them and returns as dataset """

def loadDataset(num_patches, patch_side):

    """ Load images into numpy array """

    images = scipy.io.loadmat('D:\workspace\dl\Sparse-Autoencoder\Sparse-Autoencoder-master\IMAGES.mat')
    images = images['IMAGES']
    
    """ Initialize dataset as array of zeros """
    
    dataset = numpy.zeros((patch_side*patch_side, num_patches))
    
    """ Initialize random numbers for random sampling of images
        There are 10 images of size 512 X 512 """
    
    rand = numpy.random.RandomState(int(time.time()))
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

###########################################################################################
""" Visualizes the obtained optimal W1 values as images """

def visualizeW1(opt_W1, vis_patch_side, hid_patch_side):

    """ Add the weights as a matrix of images """
    
    figure, axes = matplotlib.pyplot.subplots(nrows = hid_patch_side,
                                              ncols = hid_patch_side)
    index = 0
                                              
    for axis in axes.flat:
    
        """ Add row of weights as an image to the plot """
    
        image = axis.imshow(opt_W1[index, :].reshape(vis_patch_side, vis_patch_side),
                            cmap = matplotlib.pyplot.cm.gray, interpolation = 'nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1
        
    """ Show the obtained plot """  
        
    matplotlib.pyplot.show()

###########################################################################################
""" Loads data, trains the Autoencoder and visualizes the learned weights """

def executeSparseAutoencoder( max_iterations=1):

    """ Define the parameters of the Autoencoder """
    
    vis_patch_side = 8      # 图片采样尺寸  输入层
    hid_patch_side = 5      # 隐藏层图块尺寸  隐藏层
    rho            = 0.01   # 稀疏性参数，平均激活度
    rate          = 0.001 # 权重衰减参数，学习速度
    penalty           = 3      # 稀疏性惩罚参数
    num_patches    = 10000  # 样本个数
    max_iterations = 100    # 优化迭代字数

    visible_size = vis_patch_side ** 2  # 输入的尺寸为采样的长*款，整理长等于款
    hidden_size  = hid_patch_side **2  # 隐藏层尺寸

    #载入数据
    training_data = loadDataset(num_patches, vis_patch_side)

    #初始化自编码器参数
    encoder = SparseAutoencoder(visible_size, hidden_size, rho, rate, penalty)
    #训练
    opt_solution  = scipy.optimize.minimize(encoder.sparseAutoencoderCost, encoder.theta, 
                                            args = (training_data,), method = 'L-BFGS-B', 
                                            jac = True, options = {'maxiter': max_iterations})

    #
    opt_theta     = opt_solution.x
    opt_W1        = opt_theta[encoder.limit0 : encoder.limit1].reshape(hidden_size, visible_size)
    

    #
    visualizeW1(opt_W1, vis_patch_side, hid_patch_side)

if __name__ == "__main__":
    executeSparseAutoencoder(1)


  
    
    
    
    




