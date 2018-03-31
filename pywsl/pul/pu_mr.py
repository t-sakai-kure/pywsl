import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets

import pywsl.utils.comcalc as com
from pywsl.utils import check


class PU_SL(BaseEstimator, ClassifierMixin):

    def __init__(self, prior=.5, sigma=.1, lam=1, basis='gauss', n_basis=200):
        check.in_range(prior, 0, 1, name="prior")
        self.prior = prior
        self.basis = basis
        self.sigma = sigma
        self.lam = lam
        self.n_basis = n_basis
#            if self.sigma is None:
#                d_u = com.squared_dist(x_u, self._x_c)
#                self.sigma = np.sqrt(np.median(d_u))


    def fit(self, x, y):
        check_classification_targets(y)
#        x, y = check_X_y(x, y, y_numeric=True)
        x, y = check_X_y(x, y)
        x_p, x_u = x[y == +1, :], x[y == 0, :]
        n_p, n_u = x_p.shape[0], x_u.shape[0]

        if self.basis == 'gauss':
            b = np.minimum(n_u, self.n_basis)
            center_index = np.random.permutation(n_u)[:b]
            self._x_c = x_u[center_index, :]
        elif self.basis == 'lm':
            b = x_p.shape[1] + 1
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
        x = check_array(x)
        return np.sign(.1 + np.sign(self._ker(x).dot(self.coef_)))


    def score(self, x, y):
        x_p, x_u = x[y == +1, :], x[y == 0, :]
        k_p, k_u = self._ker(x[y == +1, :]), self._ker(x[y == 0, :])
        g_p, g_u = k_p.dot(self.coef_), k_u.dot(self.coef_)
        pu_risk = calc_risk(g_p, g_u, self.prior)
        return 1 - pu_risk


    def _ker(self, x):
        if self.basis == 'gauss':
            K = com.gauss_basis(com.squared_dist(x, self._x_c), self.sigma)
        elif self.basis == 'lm':
            K = com.homo_coord(x)

        return K


def pu_sl(x_p, x_u, prior, n_fold=5, model='gauss', sigma_list=None, 
          lambda_list=np.logspace(-3, 1, 11), n_basis=200):
    check.same_dim(x_p, x_u)

    n_p, n_u = x_p.shape[0], x_u.shape[0]

    if model == 'gauss':
        b = np.minimum(n_basis, n_u)
        center_index = np.random.permutation(n_u)[:b]
        x_c = x_u[center_index, :]
        d_p, d_u = com.squared_dist(x_p, x_c), com.squared_dist(x_u, x_c)
        if sigma_list is None:
            med = np.median(d_u.ravel())
            sigma_list = np.sqrt(med)*np.logspace(-1, 1, 11)
    elif model == 'lm':
        b, x_c = x_p.shape[0] + 1, None
        d_p, d_u = com.homo_coord(x_p), com.homo_coord(x_u)
        sigma_list = [1]
    else:
        raise ValueError('Invalid model: {} is not supported.'.format(model))

    cv_index_p, cv_index_u = com.cv_index(n_p, n_fold), com.cv_index(n_u, n_fold)

    score_cv_fold = np.empty(len(sigma_list), len(lambda_list), n_fold)

    if len(sigma_list) == 1 and len(lambda_list) == 1:
        score_cv = np.empty((1, 1))
        score_cv[0, 0] = -np.inf
        best_sigma_index, best_lambda_index = 1, 1
    else:
        for ite_sig, sigma in enumerate(sigma_list):
            K_p, K_u = ker(d_p, sigma, model), ker(d_u, sigma, model)

            for ite_fold in range(n_fold):
                K_ptr = K_p[cv_index_p != ite_fold, :]
                K_pte = K_p[cv_index_p == ite_fold, :]
                K_utr = K_u[cv_index_u != ite_fold, :]
                K_ute = K_u[cv_index_u == ite_fold, :]

                H_tr = K_utr.T.dot(K_utr)/K_u_tr.shape[0]
                h_tr = 2*prior*np.mean(K_ptr, axis=0) - np.mean(K_utr, axis=0)

                for ite_lam, lam in enumerate(lambda_list):
                    Reg = lam*np.eye(b)
                    w = np.linalg.solve(H_utr + Reg, h_tr)
                    gp, gu = K_pte.dot(w), K_ute.dot(w)
                    score_cv_fold[ite_sig, ite_lam, ite_fold] \
                        = calc_risk(gp, gu, prior)

        score_cv = np.mean(score_cv_fold, axis=2)

    score_best = np.inf
    index = np.argmin(score_cv.ravel())
    index = np.unravel_index(tmp. score_cv.shape)
    best_sigma_index, best_labmda_index = index[0], index[1]

    sigma = sigma_list[best_sigma_index]
    lam = lambda_list[best_lambda_index]
    K_p, K_u = ker(d_p, sigma, model), ker(d_u, sigma, model)
    H = K_u.T.dot(K_u)/n_u
    h = 2*prior*np.mean(K_p, axis=0) - np.mean(K_u, axis=0)
    Reg = lam*np.eye(b)
    w = np.linalg.solve(H + Reg, h)

    f_dec = lambda x_t: make_func(w, x_c, sigma, model, x_t)
    if nargout > 1:
        outs = {'sigma_index': best_sigma_index,
                'lambda_index': best_lambda_index,
                'score_cv': score_cv,
                'w': w}
        return f_dec, outs

    return f_dec


def calc_risk(gp, gu, prior):
    f_n = np.mean(gp <= 0)
    f_pu = np.mean(gu >= 0)
    pu_risk = prior*f_n + np.maximum(f_pu + prior*f_n - prior, 0)
    return pu_risk


def ker(d, sigma, model):
    if model == 'gauss':
        return com.gauss_basis(d, sigma)
    elif model == 'lm':
        return d


def make_func(w, x_c, sigma, model, x_t):
    if model == 'gauss':
        K = com.gauss_basis(com.squared_dist(x_t, x_c), sigma)
    elif model == 'lm':
        K = com.homo_coord(x_t)
    return K.dot(w)


