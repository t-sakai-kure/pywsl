import unittest

import numpy as np

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.estimator_checks import check_estimator

from pywsl.pul import pu_mr
from pywsl.utils.syndata import gen_twonorm_pu
from pywsl.utils.comcalc import bin_clf_err

class TestPU_SL(unittest.TestCase):

#    def test_check_estimator(self):
#        check_estimator(pu_mr.PU_SL)

    def test_fit(self):
        prior = .5
        x, y, x_t, y_t = gen_twonorm_pu(n_p=30, n_u=200, 
                                        prior_u=prior, n_t=100)
        pu_sl = pu_mr.PU_SL(prior, basis='lm')
        pu_sl.fit(x, y)
        y_h = pu_sl.predict(x_t)
        err = bin_clf_err(y_h, y_t, prior)
        self.assertLess(err, .2)


    def test_cv(self):
        prior = .5
        x, y, x_t, y_t = gen_twonorm_pu(n_p=30, n_u=200, 
                                        prior_u=prior, n_t=100)
        param_grid = {'prior': [prior], 
                      'lam': np.logspace(-3, 1, 5), 
                      'basis': ['lm']}
        lambda_list = np.logspace(-3, 1, 5)
        clf = GridSearchCV(estimator=pu_mr.PU_SL(), 
                           param_grid=param_grid,
                           cv=5, n_jobs=-1)
        clf.fit(x, y)
        y_h = clf.predict(x_t)
        err = bin_clf_err(y_h, y_t, prior)
        self.assertLess(err, .2)
#        print(clf.best_estimator_)
#        print(clf.cv_results_['mean_test_score'])


if __name__ == "__main__":
    unittest.main()



