import unittest

import numpy as np
import scipy as sp

from pnu_mr import pnu_sl
from pnu_mr import demo


class TestPNU_SL(unittest.TestCase):

    def test_linear_model(self):
        n_l, prior_l = 30, .3
        n_u, prior_u = 200, .5
        n_t = 100
        eta_list = np.arange(-.9, 1, .1)
        x, y, x_tp, x_tn = demo.gendata(n_l, prior_l, n_u, prior_u, n_t)
        f_dec, outs, funcs = pnu_sl.PNU_SL(x, y, prior_u, 
                                           eta_list, model='lm', 
                                           nargout=3)
        err = 100*demo.calc_err(f_dec, x_tp, x_tn, prior_u)


    def test_linear_model_one_eta(self):
        n_l, prior_l = 30, .5
        n_u, prior_u = 200, .3
        n_t = 100
        x, y, x_tp, x_tn = demo.gendata(n_l, prior_l, n_u, prior_u, n_t)
        eta_list = [pnu_sl.calc_etab(np.sum(y == +1), np.sum(y == -1), prior_u)]
        f_dec, outs, funcs = pnu_sl.PNU_SL(x, y, prior_u, 
                                           eta_list, model='lm', 
                                           nargout=3)
        err = 100*demo.calc_err(f_dec, x_tp, x_tn, prior_u)

    def test_model_without_cv(self):
        n_l, prior_l = 30, .3
        n_u, prior_u = 200, .5
        n_t = 100
        x, y, x_tp, x_tn = demo.gendata(n_l, prior_l, n_u, prior_u, n_t)
        eta_list = [pnu_sl.calc_etab(np.sum(y == +1), np.sum(y == -1), prior_u)]
        lambda_list = [.1]
        f_dec, outs, funcs = pnu_sl.PNU_SL(x, y, prior_u, 
                                           eta_list, 
                                           lambda_list=lambda_list,
                                           model='lm', nargout=3)
        err = 100*demo.calc_err(f_dec, x_tp, x_tn, prior_u)


    def test_model_nargout2(self):
        n_l, prior_l = 30, .5
        n_u, prior_u = 200, .3
        n_t = 100
        x, y, x_tp, x_tn = demo.gendata(n_l, prior_l, n_u, prior_u, n_t)
        eta_list = [pnu_sl.calc_etab(np.sum(y == +1), np.sum(y == -1), prior_u)]
        lambda_list = [.1]
        f_dec, outs = pnu_sl.PNU_SL(x, y, prior_u, 
                                    eta_list, lambda_list=lambda_list,
                                    model='lm', nargout=2)
        err = 100*demo.calc_err(f_dec, x_tp, x_tn, prior_u)


    def test_model_nargout1(self):
        n_l, prior_l = 30, .3
        n_u, prior_u = 200, .5
        n_t = 100
        x, y, x_tp, x_tn = demo.gendata(n_l, prior_l, n_u, prior_u, n_t)
        eta_list = [pnu_sl.calc_etab(np.sum(y == +1), np.sum(y == -1), prior_u)]
        lambda_list = [.1]
        f_dec = pnu_sl.PNU_SL(x, y, prior_u, eta_list, lambda_list=lambda_list,
                              model='lm', nargout=1)
        err = 100*demo.calc_err(f_dec, x_tp, x_tn, prior_u)


    def test_gauss_model(self):
        n_l, prior_l = 30, .3
        n_u, prior_u = 200, .5
        n_t = 100
        eta_list = np.arange(-.9, 1, .1)
        lambda_list = [.1]
        x, y, x_tp, x_tn = demo.gendata(n_l, prior_l, n_u, prior_u, n_t)
        f_dec, outs, funcs = pnu_sl.PNU_SL(x, y, prior_u, 
                                           eta_list, model='gauss', 
                                           lambda_list=lambda_list,
                                           nargout=3)
        err = 100*demo.calc_err(f_dec, x_tp, x_tn, prior_u)


    def test_label_specification_error(self):
        n_l, prior_l = 30, .3
        n_u, prior_u = 200, .5
        n_t = 100
        x, y, x_tp, x_tn = demo.gendata(n_l, prior_l, n_u, prior_u, n_t)
        y[0] = 2
        eta_list = [pnu_sl.calc_etab(np.sum(y == +1), np.sum(y == -1), prior_u)]
        with self.assertRaises(ValueError):
            pnu_sl.PNU_SL(x, y, prior_u, eta_list, model='lm')

    def test_model_specification_error(self):
        n_l, prior_l = 30, .3
        n_u, prior_u = 200, .5
        n_t = 100
        x, y, x_tp, x_tn = demo.gendata(n_l, prior_l, n_u, prior_u, n_t)
        eta_list = [pnu_sl.calc_etab(np.sum(y == +1), np.sum(y == -1), prior_u)]
        with self.assertRaises(ValueError):
            pnu_sl.PNU_SL(x, y, prior_u, eta_list, model='DNN')


    def test_calc_etab(self):
        n_p, n_n = 10, 10
        prior_list = np.arange(.1, 1, .1)
        for prior in prior_list:
            eta = pnu_sl.calc_etab(n_p, n_n, prior)
            if prior < .5:
                self.assertTrue(eta > 0)
            elif prior > .5:
                print(eta)
                self.assertTrue(eta < 0)
            else:
                self.assertTrue(eta == 0)



if __name__ == "__main__":
    unittest.main()
