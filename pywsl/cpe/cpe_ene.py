#!/usr/bin/env python

import numpy as np
import scipy as sp


def cpe(xl, y, xu, alpha=1):
    c = np.sort(np.unique(y))
    n_c = len(c)

    x = []
    for i in range(n_c):
        x.append(xl[y == c[i], :])
    
    b = np.empty(n_c)
    for i in range(n_c):
        b[i] = 2*np.mean(alpha_dist(x[i], xu, alpha))

    A = np.empty((n_c, n_c))
    for i in range(n_c):
        for j in range(n_c):
            A[i, j] = -np.mean(alpha_dist(x[i], x[j], alpha))

#    T3 = np.mean(alpha_dist(xu, xu, alpha))

    As = A[:(n_c-1), :(n_c-1)]
    a = A[:(n_c-1), n_c-1]
    Ac = A[n_c-1, n_c-1]
    bs = b[:(n_c-1)]
    bc = b[n_c-1]

    Anew = ((As - a[:, None].T) - a) + Ac
    Anew = (Anew + Anew.T)/2
    bnew = 2*a - 2*Ac + bs - bc

    x0 = -np.linalg.solve(2*Anew, bnew[:, None].T)
    x = np.minimum(np.maximum(0, x0), 1)
    theta = 1 - np.sum(x)

    return theta


def alpha_dist(x, c, alpha):
    n1, n2 = x.shape[0], c.shape[0]
    dist = np.sum(x**2, axis=1, keepdims=True) + \
            np.sum(c**2, axis=1, keepdims=True).T - 2*x.dot(c.T)
    
    if alpha == 1:
        return np.sqrt(np.maximum(dist, 0))
    else:
        return np.power(np.maximum(dist, 0), alpha/2)

