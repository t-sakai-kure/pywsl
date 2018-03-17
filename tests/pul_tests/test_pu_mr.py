import unittest

from sklearn.utils.estimator_checks import check_estimator

from pywsl.pul import pu_mr


class TestPU_SL(unittest.TestCase):

    def test_check_estimator(self):
        check_estimator(pu_mr.PU_SL(prior=0.5))

    def test_fit(self):
        pass



if __name__ == "__main__":
    unittest.main()



