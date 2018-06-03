import unittest

import numpy as np
import scipy as sp

from pywsl.ssl import pnu_mr


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


class TestPNU_SL(unittest.TestCase):


    def test_linear_model(self):
        n_l, prior_l = 30, .3
        n_u, prior_u = 200, .5
        n_t = 100
        eta_list = np.arange(-.9, 1, .1)
        x, y, x_tp, x_tn = gendata(n_l, prior_l, n_u, prior_u, n_t)
        f_dec, outs, funcs = pnu_mr.PNU_SL_FastCV(x, y, prior_u, 
                                           eta_list, basis='lm', 
                                           nargout=3)
        err = 100*calc_err(f_dec, x_tp, x_tn, prior_u)


    def test_linear_model_one_eta(self):
        n_l, prior_l = 30, .5
        n_u, prior_u = 200, .3
        n_t = 100
        x, y, x_tp, x_tn = gendata(n_l, prior_l, n_u, prior_u, n_t)
        eta_list = [pnu_mr.calc_etab(np.sum(y == +1), np.sum(y == -1), prior_u)]
        f_dec, outs, funcs = pnu_mr.PNU_SL_FastCV(x, y, prior_u, 
                                           eta_list, basis='lm', 
                                           nargout=3)
        err = 100*calc_err(f_dec, x_tp, x_tn, prior_u)

    def test_model_without_cv(self):
        n_l, prior_l = 30, .3
        n_u, prior_u = 200, .5
        n_t = 100
        x, y, x_tp, x_tn = gendata(n_l, prior_l, n_u, prior_u, n_t)
        eta_list = [pnu_mr.calc_etab(np.sum(y == +1), np.sum(y == -1), prior_u)]
        lambda_list = [.1]
        f_dec, outs, funcs = pnu_mr.PNU_SL_FastCV(x, y, prior_u, 
                                           eta_list, 
                                           lambda_list=lambda_list,
                                           basis='lm', nargout=3)
        err = 100*calc_err(f_dec, x_tp, x_tn, prior_u)


    def test_model_nargout2(self):
        n_l, prior_l = 30, .5
        n_u, prior_u = 200, .3
        n_t = 100
        x, y, x_tp, x_tn = gendata(n_l, prior_l, n_u, prior_u, n_t)
        eta_list = [pnu_mr.calc_etab(np.sum(y == +1), np.sum(y == -1), prior_u)]
        lambda_list = [.1]
        f_dec, outs = pnu_mr.PNU_SL_FastCV(x, y, prior_u, 
                                           eta_list, lambda_list=lambda_list,
                                           basis='lm', nargout=2)
        err = 100*calc_err(f_dec, x_tp, x_tn, prior_u)


    def test_model_nargout1(self):
        n_l, prior_l = 30, .3
        n_u, prior_u = 200, .5
        n_t = 100
        x, y, x_tp, x_tn = gendata(n_l, prior_l, n_u, prior_u, n_t)
        eta_list = [pnu_mr.calc_etab(np.sum(y == +1), np.sum(y == -1), prior_u)]
        lambda_list = [.1]
        f_dec = pnu_mr.PNU_SL_FastCV(x, y, prior_u, eta_list, lambda_list=lambda_list,
                                     basis='lm', nargout=1)
        err = 100*calc_err(f_dec, x_tp, x_tn, prior_u)


    def test_gauss_model(self):
        n_l, prior_l = 30, .3
        n_u, prior_u = 200, .5
        n_t = 100
        lambda_list = [.1]
        x, y, x_tp, x_tn = gendata(n_l, prior_l, n_u, prior_u, n_t)
        eta_list = [pnu_mr.calc_etab(np.sum(y == +1), np.sum(y == -1), prior_u)]
        f_dec, outs, funcs = pnu_mr.PNU_SL_FastCV(x, y, prior_u, 
                                           eta_list, basis='gauss', 
                                           lambda_list=lambda_list,
                                           nargout=3)
        err = 100*calc_err(f_dec, x_tp, x_tn, prior_u)


    def test_label_specification_error(self):
        n_l, prior_l = 30, .3
        n_u, prior_u = 200, .5
        n_t = 100
        x, y, x_tp, x_tn = gendata(n_l, prior_l, n_u, prior_u, n_t)
        y[0] = 2
        eta_list = [pnu_mr.calc_etab(np.sum(y == +1), np.sum(y == -1), prior_u)]
        with self.assertRaises(ValueError):
            pnu_mr.PNU_SL_FastCV(x, y, prior_u, eta_list, basis='lm')

    def test_model_specification_error(self):
        n_l, prior_l = 30, .3
        n_u, prior_u = 200, .5
        n_t = 100
        x, y, x_tp, x_tn = gendata(n_l, prior_l, n_u, prior_u, n_t)
        eta_list = [pnu_mr.calc_etab(np.sum(y == +1), np.sum(y == -1), prior_u)]
        with self.assertRaises(ValueError):
            pnu_mr.PNU_SL_FastCV(x, y, prior_u, eta_list, basis='DNN')


    def test_calc_etab(self):
        n_p, n_n = 10, 10
        prior_list = np.arange(.1, 1, .1)
        for prior in prior_list:
            eta = pnu_mr.calc_etab(n_p, n_n, prior)
            if prior < .5:
                self.assertTrue(eta > 0)
            elif prior > .5:
                self.assertTrue(eta < 0)
            else:
                self.assertTrue(eta == 0)



if __name__ == "__main__":
    unittest.main()
