import numpy as np
import scipy as sp
import unittest

from pywsl.cpe import cpe_ene
from pywsl.utils.syndata import gen_twonorm_ssl

class TestCpe_Ene(unittest.TestCase):

    def test_alpha_one(self):
        prior = .3
        x, y, x_t, y_t = gen_twonorm_ssl(n_l=100, prior_l=.5, 
                                         n_u=300, prior_u=prior,
                                         n_t=100)
        x_l = x[y != 0, :]
        y_l = y[y != 0]
        x_u = x[y == 0, :]
        priorh = cpe_ene.cpe(x_l, y_l, x_u)
        prior_lower, prior_upper = prior - .1, prior + .1
        self.assertTrue(prior_lower < priorh and priorh < prior_upper)

    def test_alpha_half(self):
        prior = .3
        x, y, x_t, y_t = gen_twonorm_ssl(n_l=100, prior_l=.5, 
                                         n_u=300, prior_u=prior,
                                         n_t=100)
        x_l = x[y != 0, :]
        y_l = y[y != 0]
        x_u = x[y == 0, :]
        priorh = cpe_ene.cpe(x_l, y_l, x_u, alpha=.5)
        prior_lower, prior_upper = prior - .1, prior + .1
        print(priorh)


if __name__ == "__main__":
    unittest.main()


