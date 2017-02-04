# -*- coding: utf-8 -*-
import scipy
import scipy.io
from NN import *
import time
import matplotlib.pyplot as plt
import matplotlib

def normalizeDataset(dataset):

    """ Remove mean of dataset """

    dataset = dataset - np.mean(dataset)

    """ Truncate to +/-3 standard deviations and scale to -1 to 1 """

    std_dev = 3 * np.std(dataset)
    dataset = np.maximum(np.minimum(dataset, std_dev), -std_dev) / std_dev

    """ Rescale from [-1, 1] to [0.1, 0.9] """

    dataset = (dataset + 1) * 0.4 + 0.1

    return dataset


def loadDataset(num_patches, patch_side):
    """ Load images into np array """

    images = scipy.io.loadmat('data/IMAGES.mat')
    images = images['IMAGES']

    """ Initialize dataset as array of zeros """

    dataset = np.zeros((patch_side * patch_side, num_patches))

    """ Initialize random numbers for random sampling of images
        There are 10 images of size 512 X 512 """

    rand = np.random.RandomState(1)




    image_indices = rand.randint(512 - patch_side, size=(num_patches, 2))
    image_number = rand.randint(10, size=num_patches)

    """ Sample 'num_patches' random image patches """

    for i in xrange(num_patches):
        """ Initialize indices for patch extraction """

        index1 = image_indices[i, 0]
        index2 = image_indices[i, 1]
        index3 = image_number[i]

        """ Extract patch and store it as a column """

        patch = images[index1:index1 + patch_side, index2:index2 + patch_side, index3]
        patch = patch.flatten()
        dataset[:, i] = patch

    """ Normalize and return the dataset """

    dataset = normalizeDataset(dataset)
    return dataset

def display_network(A, filename='weights.png'):
    opt_normalize = True
    opt_graycolor = True

    # Rescale
    A = A - np.average(A)

    # Compute rows & cols
    (row, col) = A.shape
    sz = int(np.ceil(np.sqrt(row)))
    buf = 1
    n = np.ceil(np.sqrt(col))
    m = np.ceil(col / n)

    image = np.ones(shape=(buf + m * (sz + buf), buf + n * (sz + buf)))

    if not opt_graycolor:
        image *= 0.1

    k = 0
    for i in range(int(m)):
        for j in range(int(n)):
            if k >= col:
                continue

            clim = np.max(np.abs(A[:, k]))

            if opt_normalize:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / clim
            else:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / np.max(np.abs(A))
            k += 1

    plt.imsave(filename, image, cmap=matplotlib.cm.gray)


if __name__ == '__main__':
    datas = loadDataset(1000, 8)

    vis_patch_side = 8
    hid_patch_side = 5

    encoder = Encoder(datas, vis_patch_side=8, hid_patch_side=5, train_count=10000, rate=0.0001, reg=0.0001,  sparse=0.01, punish=3.0,
                 patience_n=None, patience_v=None)
    encoder.train(100)

    print 'train end'

    display_network(encoder.w1)













'''
    pic = pic.reshape((8,8))

    print pic.shape


    plt.imshow(pic)
    plt.show()
'''



