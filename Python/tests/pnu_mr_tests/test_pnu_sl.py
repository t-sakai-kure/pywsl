import unittest

import numpy as np
import scipy as sp

from pnu_mr import pnu_sl
from pnu_mr import demo


class TestPNU_SL(unittest.TestCase):

    def test_linear_model(slef):
        n_l, prior_l = 30, .3
        n_u, prior_u = 200, .5
        n_t = 100
        eta_list = np.arange(-.9, 1, .1)
        x, y, x_tp, x_tn = demo.gendata(n_l, prior_l, n_u, prior_u, n_t)
        f_dec, outs, funcs = pnu_sl.PNU_SL(x, y, prior_u, 
                                           eta_list, model='lm', 
                                           nargout=3)
        err = 100*demo.calc_err(f_dec, x_tp, x_tn, prior_u)


if __name__ == "__main__":
    unittest.main()
