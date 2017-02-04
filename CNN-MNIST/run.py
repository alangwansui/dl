import os
import sys, getopt
import time
import numpy as np
from cnn_training_computation import fit, predict

def run():
    # read the data, labels
    data = np.genfromtxt("mnist/train-images.idx3-ubyte")
    print ". .",
    test_data = np.genfromtxt("data/mnist_test.data")
    print ". .",
    valid_data = np.genfromtxt("data/mnist_valid.data")
    labels = np.genfromtxt("data/mnist_train.solution")
    print ". . finished reading"
   
    # DO argmax
    labels = np.argmax(labels, axis=1)
    print labels
        
    # normalization
    amean = np.mean(data)
    data = data - amean
    astd = np.std(data)
    data = data / astd
    # normalise using coefficients from training data
    test_data = (test_data - amean) / astd
    valid_data = (valid_data - amean) / astd
    
    fit(data, labels)
    
    print "finished training"
    rv = predict(valid_data)
    rt = predict(test_data)



    # UNDO argmax and save results x 2
    r = rv
    N = len(r)
    res = np.zeros((N, 10))
    for i in range(N):
        res[i][r[i]] = 1
    
    np.savetxt("mnist_valid.predict", res, fmt='%i')
    
    r = rt
    N = len(r)
    res = np.zeros((N, 10))
    for i in range(N):
        res[i][r[i]] = 1
    
    np.savetxt("mnist_test.predict", res, fmt='%i')
    print "finished predicting."
   


if __name__ == '__main__':
    run()

    
    
    
    
    
    

    
    
    
    
    
    













