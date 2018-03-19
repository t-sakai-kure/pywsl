import numpy as np


def gen_twonorm_ssl(n_l, prior_l, n_u, prior_u, n_t):
    d = 2
    mu_p, mu_n = np.array([1, 1]), np.array([-1, -1])

    n_p = np.random.binomial(n_l, prior_l)
    n_n = n_l - n_p
    x_p = np.random.randn(n_p, 2) + mu_p
    x_n = np.random.randn(n_n, 2) + mu_n

    n_up = np.random.binomial(n_u, prior_u)
    n_un = n_u - n_up
    x_up = np.random.randn(n_up, 2) + mu_p
    x_un = np.random.randn(n_un, 2) + mu_n

    x = np.r_[x_p, x_n, x_up, x_un]
    y = np.r_[np.ones(n_p), -np.ones(n_n), np.zeros(n_u)]

    x_tp = np.random.randn(n_t, 2) + mu_p
    x_tn = np.random.randn(n_t, 2) + mu_n

    x_t = np.r_[x_tp, x_tn]
    y_t = np.r_[np.ones(n_t), -np.ones(n_t)]

    return x, y, x_t, y_t


def gen_twonorm_pu(n_p, n_u, prior_u, n_t):
    d = 2
    mu_p, mu_n = np.array([1, 1]), np.array([-1, -1])

    x_p = np.random.randn(n_p, 2) + mu_p

    n_up = np.random.binomial(n_u, prior_u)
    n_un = n_u - n_up
    x_up = np.random.randn(n_up, 2) + mu_p
    x_un = np.random.randn(n_un, 2) + mu_n

    x = np.r_[x_p, x_up, x_un]
    y = np.r_[np.ones(n_p), np.zeros(n_u)]

    x_tp = np.random.randn(n_t, 2) + mu_p
    x_tn = np.random.randn(n_t, 2) + mu_n

    x_t = np.r_[x_tp, x_tn]
    y_t = np.r_[np.ones(n_t), -np.ones(n_t)]

    return x, y, x_t, y_t


