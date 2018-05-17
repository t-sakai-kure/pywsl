"""
Common calculation.
"""

import numpy as np


#import check


def cv_index(n, n_fold):
    index = (np.arange(n, dtype=np.int)*n_fold)//n
    return index[np.random.permutation(n)]


def squared_dist(x, c):
#    assert x.shape[1] == c.shape[1], "Dimension must be the same."
    return np.sum(x**2, axis=1, keepdims=True) \
        + np.sum(c**2, axis=1, keepdims=True).T \
        - 2*x.dot(c.T)


def gauss_basis(dist2, sigma):
    return np.exp(-dist2/(2*sigma**2))


def homo_coord(x):
    return np.c_[x, np.ones(x.shape[0])]


def bin_clf_err(y_h, y_t, prior=None):
    # check y_t
    if prior:
        f_p = np.mean(y_h[y_t == +1] <= 0)
        f_n = np.mean(y_h[y_t == -1] >= 0)
        err = prior*f_p + (1-prior)*f_n
    else:
        err = np.mean(y_h*y_t <= 0)

    return err




