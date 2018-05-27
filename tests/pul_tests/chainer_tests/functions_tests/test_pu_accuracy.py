import unittest

import numpy as np

from pywsl.pul.chainer.functions.pu_accuracy import PU_Accuracy

class TestPU_Accuracy(unittest.TestCase):

    def setUp(self):
        self.n_p, self.n_u, self.prior = 20, 80, .5
        self.y = np.r_[np.ones(self.n_p), np.zeros(self.n_u)].astype(np.int32)
        self.fun_pu_acc = PU_Accuracy(self.prior)
        
    def test_acc_ones(self):
        yh = np.ones(self.n_p + self.n_u).astype(np.float32)
        acc = self.fun_pu_acc(yh, self.y).array
        self.assertAlmostEqual(acc, .5)

    def test_acc_minus_ones(self):
        yh = -np.ones(self.n_p + self.n_u).astype(np.float32)
        acc = self.fun_pu_acc(yh, self.y).array
        self.assertAlmostEqual(acc, .5)

    def test_acc_correct(self):
        n_up = int(self.n_u*self.prior)
        n_un = self.n_u - n_up
        yh = np.r_[np.ones(self.n_p + n_up), -np.ones(n_un)].astype(np.float32)
        acc = self.fun_pu_acc(yh, self.y).array
        self.assertAlmostEqual(acc, 1.)


if __name__ == "__main__":
    unittest.main()
