#!/usr/bin/env python

import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2.0, rc={"lines.linewidth": 4.0})
sns.set_style('ticks')


import pywsl.ssl.pnu_mr as pnu
import pywsl.cpe.cpe_ene as cpe


def calc_err(f_dec, x_tp, x_tn, prior):
    g_p, g_n = f_dec(x_tp), f_dec(x_tn)
    return prior*np.mean(g_p <= 0) + (1-prior)*np.mean(g_n >= 0)


def gendata(n_l, prior_l, n_u, prior_u, n_t):
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

    return x, y, x_tp, x_tn


if __name__ == "__main__":
    np.random.seed(1)

    n_l = 10
    n_u = 300
    n_t = 1000

    prior_l = .5
    prior_u = .3

    eta_list = np.arange(-.9, 1, .1)

    n_trial = 20

    best_err = np.inf
    errs1 = np.empty(n_trial)
    errs2 = np.empty((n_trial, len(eta_list)))
    priors = np.empty(n_trial)
    for ite in range(n_trial):
        x, y, x_tp, x_tn = gendata(n_l, prior_l, n_u, prior_u, n_t)
        priorh = cpe.cpe(x[y != 0, :], y[y != 0], x[y == 0, :])
        f_dec, outs, funcs = pnu.PNU_SL_FastCV(x, y, priorh, eta_list, 
                                               lambda_list=[.1], 
                                               basis='lm', nargout=3)
        errs1[ite] = 100*calc_err(f_dec, x_tp, x_tn, prior_u)
        if errs1[ite] < best_err:
            best_err = errs1[ite]
            best_w = outs['w']
            best_x, best_y = x, y

        for ite_eta in range(len(eta_list)):
            errs2[ite, ite_eta] = 100*calc_err(funcs[ite_eta],
                                               x_tp, x_tn, prior_u)
        priors[ite] = priorh

    print("Average of misclassification rates: {:.1f} ({:.2f})".format(
        np.mean(errs1), np.std(errs1)/np.sqrt(n_trial)))
    print("Average of estimated class-priors: {:.2f} ({:.2f})".format(
        np.mean(priors), np.std(priors)/np.sqrt(n_trial)))


    x_p = best_x[best_y == +1, :]
    x_n = best_x[best_y == -1, :]
    x_u = best_x[best_y ==  0, :]


    fig1 = plt.figure(1)
    plt.scatter(x_u[:, 0], x_u[:, 1], c='k', marker='.')
    plt.scatter(x_p[:, 0], x_p[:, 1], facecolors='none', edgecolor='b',
                marker='o', s=300, lw=5)
    plt.scatter(x_n[:, 0], x_n[:, 1], c='r',
                marker='x', s=300, lw=5)

    u1, u2 = np.min(x[:, 0]), np.max(x[:, 0])
    v1_opt = np.log(prior_u/(1-prior_u))/2 - u1
    v2_opt = np.log(prior_u/(1-prior_u))/2 - u2
    plt.plot([u1, u2], [v1_opt, v2_opt])

    w = best_w[:2]
    intercept = best_w[2]
    v1_est = (intercept - w[0]*u1)/w[1]
    v2_est = (intercept - w[0]*u2)/w[1]
    plt.plot([u1, u2], [v1_est, v2_est])
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    sns.despine()
    plt.savefig('data.pdf')
    
    ###
    fig2 = plt.figure(2)
    plt.errorbar(eta_list, np.mean(errs2, axis=0),
                 yerr=np.std(errs2, axis=0)/np.sqrt(n_trial))
    sns.despine()
    plt.savefig('err_curve.pdf')

    plt.show()
    
