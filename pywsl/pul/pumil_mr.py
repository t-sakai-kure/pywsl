import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets

import pywsl.utils.comcalc as com
from pywsl.utils import check
from pywsl.utils.check import check_bags

from pywsl.pul.pu_mr import calc_risk


class PUMIL_SL(BaseEstimator, ClassifierMixin):

    def __init__(self, prior=.5, degree=1, lam=1, basis='minimax', n_basis=200):
        check.in_range(prior, 0, 1, name="prior")
        self.prior = prior
        self.basis = basis
        self.degree = degree
        self.lam = lam
        self.n_basis = n_basis


    def fit(self, x, y):
        check_classification_targets(y)
        x = check_bags(x)
        check_consistent_length(x, y)
        x_p = [x[i] for i in np.where(y == +1)[0]]
        x_u = [x[i] for i in np.where(y == +0)[0]]
        n_p, n_u = len(x_p), len(x_u)

        if self.basis == 'minimax':
            b = np.minimum(n_u, self.n_basis)
            center_index = np.random.permutation(n_u)[:b]
            self._x_c = [x_u[i] for i in center_index]
        else:
            raise ValueError('Invalid basis type: {}.'.format(basis))

        k_p, k_u = self._ker(x_p), self._ker(x_u)

        H = k_u.T.dot(k_u)/n_u
        h = 2*self.prior*np.mean(k_p, axis=0) - np.mean(k_u, axis=0)
        R = self.lam*np.eye(b)
        self.coef_ = sp.linalg.solve(H + R, h)

        return self


    def predict(self, x):
        check_is_fitted(self, 'coef_')
        x = check_bags(x)
        return np.sign(.1 + np.sign(self._ker(x).dot(self.coef_)))


    def score(self, x, y):
        x_p = [x[i] for i in np.where(y == +1)[0]]
        x_u = [x[i] for i in np.where(y == +0)[0]]
        k_p, k_u = self._ker(x_p), self._ker(x_u)
        g_p, g_u = k_p.dot(self.coef_), k_u.dot(self.coef_)
        pu_risk = calc_risk(g_p, g_u, self.prior)
        return 1 - pu_risk


    def _ker(self, x):
        if self.basis == 'minimax':
            # minimax polynomial kernel (Andrews et al., NIPS2002)
            stat = lambda X: np.concatenate([X.max(axis=0), X.min(axis=0)])
            sx = np.array([stat(X) for X in x])
            sc = np.array([stat(X) for X in self._x_c])
            K = polynomial_kernel(sx, sc, degree=self.degree)

        return K
