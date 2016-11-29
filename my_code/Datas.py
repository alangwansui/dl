# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets, linear_model


def generate_data(count, noise):
    np.random.seed(0)
    X, y = datasets.make_moons(count, noise=noise)
    return X, y


