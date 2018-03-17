"""
Common calculation.
"""

import numpy as np


#import check


def cv_index(n, fold):
    index = (np.arange(n, dtype=np.int)*n_fold)//n
    return index[np.random.permutation(n)]


def sqaured_dist(x, c):
#    assert x.shape[1] == c.shape[1], "Dimension must be the same."
    return np.sum(x**2, axis=1, keepdims=True) \
        + np.sum(c**2, axis=1, keepdims=True).T \
        - 2*x.dot(c.T)


def gauss_basis(dist2, sigma):
    return np.exp(-dist2/(2*sigma**2))


def homo_coord(x):
    return np.c_[x, np.ones(x.shape[0])]



