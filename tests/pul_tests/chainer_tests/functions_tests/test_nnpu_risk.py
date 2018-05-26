import unittest

import numpy as np
import chainer.functions as F

from pywsl.pul.chainer.functions.nnpu_risk import PU_Risk


class TestPU_Risk(unittest.TestCase):

    def setUp(self):
        self.n_p, self.n_u, self.prior = 20, 80, .5
        self.y = np.r_[np.ones(self.n_p), np.zeros(self.n_u)].astype(np.int32)

    def test_nnpu(self):
        yh = np.r_[2*np.ones(self.n_p), -np.ones(self.n_u)].astype(np.float32)[:, None]
        fun_pu_risk = PU_Risk(self.prior, nnPU=False)
        pu_risk = fun_pu_risk(yh, self.y).array
        self.assertLess(pu_risk, 0)
        fun_nnpu_risk = PU_Risk(self.prior, nnPU=True)
        nnpu_risk = fun_nnpu_risk(yh, self.y).array
        self.assertGreater(nnpu_risk, 0)


if __name__ == "__main__":
    unittest.main()
