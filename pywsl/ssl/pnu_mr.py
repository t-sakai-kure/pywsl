import numpy as np
import scipy as sp
from functools import partial

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

import pywsl.utils.comcalc as com
from pywsl.utils import check


class PNU_SL(BaseEstimator, ClassifierMixin):

    def __init__(self, prior, eta=.5, sigma=.1, lam=1, basis='gauss', n_basis=200):
        check.in_range(prior, 0, 1, name='prior')
        check.in_range(eta, -1, 1, name='combination parameter')
        self.prior = prior
        self.basis = basis
        self.eta = eta
        self.sigma = sigma
        self.lam = lam
        self.n_basis = n_basis

    def fit(self, x, y):
        x, y = check_X_y(x, y)
        if not np.array_equal(np.unique(y), [-1, 0, +1]):
            raise ValueError("""Label vector is invalid.
            Expected: [-1, 0, +1] == np.unique(y)
            Actual: {} == np.unique(y)
            """.format(np.unique(y)))
        x_p, x_n, x_u = x[y == +1, :], x[y == -1, :], x[y == 0, :]
        n_p, n_n, n_u = x_p.shape[0], x_n.shape[0], x_u.shape[0]

        if self.basis == 'gauss':
            b = np.minimum(n_u, self.n_basis)
            center_index = np.random.permutation(n_u)[:b]
            self.x_c = x_u[center_index, :]
        elif self.basis == 'lm':
            b = x_p.shape[1] + 1
        else:
            raise ValueError('Invalid basis type: {}.'.format(basis))

        k_p, k_n, k_u = self._ker(x_p), self._ker(x_n), self._ker(x_u)
        th_p, th_n = self.prior, 1-self.prior

        H_pn = th_p*(k_p.T.dot(k_p))/n_p + th_n*(k_n.T.dot(k_n))/n_n
        H_u = k_u.T.dot(k_u)/n_u
        h_p = th_p*np.sum(k_p, axis=0)/n_p
        h_n = th_n*np.sum(k_n, axis=0)/n_n
        h_u = np.sum(k_u, axis=0)/n_u

        h_pn = h_p - h_n
        Reg = self.lam*np.eye(b)
        if self.basis == 'lm':
            Reg[b-1, b-1] = 0
        if self.eta >= 0: # PN+PU
            eta = self.eta
            h_xu = 2*h_p - h_u
        else: # PN+NU
            eta = -self.eta
            h_xu = h_u - 2*h_n

        self.etah_ = calc_etab(n_p, n_n, th_p)
        self.coef_ = sp.linalg.solve((1-eta)*H_pn + eta*H_u + Reg,
                                     (1-eta)*h_pn + eta*h_xu)
        return self
    
    def predict(self, x):
        check_is_fitted(self, 'coef_')
        x = check_array(x)
        return np.sign(.1 + np.sign(self._ker(x).dot(self.coef_)))

    def _ker(self, x):
        if self.basis == 'gauss':
            K = com.gauss_basis(com.squared_dist(x, self.x_c), self.sigma)
        elif self.basis == 'lm':
            K = com.homo_coord(x)
        return K


def pnu_risk(y_true, y_pred, prior, eta):
    yh_p, yh_n, yh_u = y_pred[y_true == +1], y_pred[y_true == -1], y_pred[y_true == 0]
    n_p, n_n, n_u = max([1, len(yh_p)]), max([1, len(yh_n)]), max([1, len(yh_u)])
    th_p, th_n = prior, 1-prior
    false_neg = np.sum(yh_p == -1)/n_p
    false_pos = np.sum(yh_n == +1)/n_n
    pn_risk = th_p*false_neg + th_n*false_pos
    if eta >= 0: # PN+PU
        false_pos_u = np.sum(yh_u == +1)/n_u
        xu_risk = th_p*false_neg + np.maximum(0, false_pos_u + th_p*false_neg - th_p)
    else: # PN+NU
        eta = -eta
        false_neg_u = np.sum(yh_u == -1)/n_u
        xu_risk = th_n*false_pos + np.maximum(0, false_neg_u + th_n*false_pos - th_n)
    risk = (1-eta)*pn_risk + eta*xu_risk
    return 1 - risk


def PNU_SL_FastCV(x, y, prior, eta_list, n_fold=5, basis='gauss',
                  sigma_list=None, lambda_list=np.logspace(-3, 1, 10),
                  n_basis=200, nargout=2):

    if not np.array_equal(np.unique(y), [-1, 0, +1]):
        raise ValueError("""Label vector is invalid.
        Expected: [-1, 0, +1] == np.unique(y)
        Actual: {} == np.unique(y)
        """.format(np.unique(y)))

    x_p, x_n, x_u = x[y == +1, :], x[y == -1, :], x[y ==  0, :]
    n_p, n_n, n_u = x_p.shape[0], x_n.shape[0], x_u.shape[0]

    if basis == 'gauss':
        b = np.minimum(n_basis, n_u)
        center_index = np.random.permutation(n_u)[:b]
        x_c = x_u[center_index, :]
        d_p, d_n, d_u = sqdist(x_p, x_c), sqdist(x_n, x_c), sqdist(x_u, x_c)
        if sigma_list is None:
            med = np.median(d_u.ravel())
            sigma_list = np.sqrt(med)*np.logspace(-1, 1, 11)
    elif basis == 'lm':
        b = x.shape[1] + 1
        x_c = None
        d_p, d_n, d_u = np.c_[x_p, np.ones(n_p)], np.c_[x_n, np.ones(n_n)], \
                        np.c_[x_u, np.ones(n_u)]
        sigma_list = [1]
    else:
        raise ValueError('Invalid basis type.')

    cv_index_p = (np.arange(n_p, dtype=np.int_)*n_fold)//n_p
    cv_index_p = cv_index_p[np.random.permutation(n_p)]
    cv_index_n = (np.arange(n_n, dtype=np.int_)*n_fold)//n_n
    cv_index_n = cv_index_n[np.random.permutation(n_n)]
    cv_index_u = (np.arange(n_u, dtype=np.int_)*n_fold)//n_u
    cv_index_u = cv_index_u[np.random.permutation(n_u)]

    etab = calc_etab(n_p, n_n, prior)
    
    score_cv_fold = np.empty((len(sigma_list), len(lambda_list),
                              len(eta_list), n_fold))
    if len(eta_list) == 1 and len(sigma_list) == 1 and \
       len(lambda_list) == 1:
        score_cv = np.empty((1, 1, 1))
        score_cv[0, 0, 0] = -np.inf
        best_sigma_index, best_lambda_index, best_eta_index \
            = 1, 1, 1
    else:
        for ite_sigma, sigma in enumerate(sigma_list):
            K_p, K_n, K_u = ker(d_p, d_n, d_u, sigma, basis)

            for ite_fold in range(n_fold):
                H_ptr, H_ntr, H_utr, h_ptr, h_ntr, h_utr = make_mat(
                    K_p[cv_index_p != ite_fold, :],
                    K_n[cv_index_n != ite_fold, :],
                    K_u[cv_index_u != ite_fold, :], prior)

                K_pte = K_p[cv_index_p == ite_fold, :]
                K_nte = K_n[cv_index_n == ite_fold, :]
                K_ute = K_u[cv_index_u == ite_fold, :]

                for ite_eta, eta in enumerate(eta_list):
                    for ite_lambda, lam in enumerate(lambda_list):
                        theta_cv = solve(H_ptr, H_ntr, H_utr, h_ptr, h_ntr, h_utr,
                                         lam, eta, basis, b)
                        gp, gn, gu = K_pte.dot(theta_cv), K_nte.dot(theta_cv), \
                                     K_ute.dot(theta_cv)
                        score_cv_fold[ite_sigma, ite_lambda, ite_eta, ite_fold] \
                            = calc_risk(gp, gn, gu, prior, etab)

        score_cv = np.mean(score_cv_fold, axis=3)

    score_list = np.empty(len(eta_list))
    score_best = np.inf
    funcs = []
    for ite_eta, eta in enumerate(eta_list):
        sub_score_cv = score_cv[:, :, ite_eta]
        tmp = np.argmin(sub_score_cv.ravel())
        tmp = np.unravel_index(tmp, sub_score_cv.shape)
        sigma_index, lambda_index = tmp[0], tmp[1]
        score_list[ite_eta] = sub_score_cv[sigma_index, lambda_index]
        if nargout == 3:
            sigma, lam = sigma_list[sigma_index], lambda_list[lambda_index]
            K_p, K_n, K_u = ker(d_p, d_n, d_u, sigma, basis)
            H_p, H_n, H_u, h_p, h_n, h_u = make_mat(K_p, K_n, K_u, prior)
            theta = solve(H_p, H_n, H_u, h_p, h_n, h_u, lam, eta, basis, b)
            funcs.append(partial(make_func, theta, x_c, sigma, basis))
        if score_list[ite_eta] < score_best:
            score_best = score_list[ite_eta]
            best_sigma_index = sigma_index
            best_lambda_index = lambda_index
            best_eta_index  = ite_eta
            if nargout > 2:
                best_theta = theta

    sigma, lam = sigma_list[best_sigma_index], lambda_list[best_lambda_index]
    eta = eta_list[best_eta_index]

    if nargout > 1:
        outs = {'sigma_index': best_sigma_index,
                'lambda_index': best_lambda_index,
                'eta_index': best_eta_index,
                'score_table': score_cv,
                'score_list': score_list}
    if nargout < 3:
        K_p, K_n, K_u = ker(d_p, d_n, d_u, sigma, basis)
        H_p, H_n, H_u, h_p, h_n, h_u = make_mat(K_p, K_n, K_u, prior)
        theta = solve(H_p, H_n, H_u, h_p, h_n, h_u, lam, eta, basis, b)
        f_dec = partial(make_func, theta, x_c, sigma, basis)
        if nargout == 1:
            return f_dec
        else:
            outs['w'] = theta
            return f_dec, outs
    if nargout > 2:
        f_dec = funcs[best_eta_index]
        outs['w'] = best_theta
        return f_dec, outs, funcs


def make_mat(K_p, K_n, K_u, prior):
    n_p, n_n, n_u = K_p.shape[0], K_n.shape[0], K_u.shape[0]
    H_p = prior*(K_p.T.dot(K_p))/n_p if n_p != 0 else 0
    H_n = (1-prior)*(K_n.T.dot(K_n))/n_n if n_n != 0 else 0
    H_u = (K_u.T.dot(K_u))/n_u if n_u != 0 else 0
    h_p = prior*np.mean(K_p, axis=0) if n_p != 0 else 0
    h_n = (1-prior)*np.mean(K_n, axis=0) if n_n != 0 else 0
    h_u = np.mean(K_u, axis=0) if n_u != 0 else 0
    return H_p, H_n, H_u, h_p, h_n, h_u


def solve(H_p, H_n, H_u, h_p, h_n, h_u, lam, eta, basis, b):
    Reg = lam*np.eye(b)
    if basis == 'lm':
        Reg[b-1, b-1] = 0

    H_pn = H_p + H_n
    h_pn = h_p - h_n

    if eta >= 0: # PNPU
        h_pu = 2*h_p - h_u
        theta = np.linalg.solve((1-eta)*H_pn + eta*H_u + Reg,
                                (1-eta)*h_pn + eta*h_pu)
    else: # PNNU
        eta = -eta
        h_nu = h_u - 2*h_n
        theta = np.linalg.solve((1-eta)*H_pn + eta*H_u + Reg,
                                (1-eta)*h_pn + eta*h_nu)
    return theta


def calc_risk(g_p, g_n, g_u, prior, eta):
    n_p, n_n, n_u = g_p.shape[0], g_n.shape[0], g_u.shape[0]

    f_n = np.mean(g_p <= 0) if n_p != 0 else 0
    f_p = np.mean(g_n >= 0) if n_n != 0 else 0
    risk_pn = prior*f_n + (1-prior)*f_p

    if eta >= 0: # PNPU
        f_pu = np.mean(g_u >= 0) if n_u != 0 else 0
        risk_pu = prior*f_n + np.maximum(f_pu + prior*f_n - prior, 0)
        risk = (1-eta)*risk_pn + eta*risk_pu
    else: # PNNU
        eta = -eta
        f_nu = np.mean(g_u <= 0) if n_u != 0 else 0
        risk_nu = np.maximum(f_nu + (1-prior)*f_p - (1-prior), 0) + (1-prior)*f_p
        risk = (1-eta)*risk_pn + eta*risk_nu

    return risk


def calc_etab(n_p, n_n, prior):
    psi_p, psi_n = prior**2/n_p, (1-prior)**2/n_n
    return (psi_n - psi_p)/(psi_p + psi_n)


def ker(d_p, d_n, d_u, sigma, basis):
    if basis == 'gauss':
        K_p = np.exp(-d_p/(2*sigma**2))
        K_n = np.exp(-d_n/(2*sigma**2))
        K_u = np.exp(-d_u/(2*sigma**2))
    elif basis == 'lm':
        K_p, K_n, K_u = d_p, d_n, d_u

    return K_p, K_n, K_u


def sqdist(x, c):
    n1, n2 = x.shape[0], c.shape[0]
    return np.sum(x**2, axis=1, keepdims=True) \
        + np.sum(c**2, axis=1, keepdims=True).T \
        - 2*x.dot(c.T)


def make_func(theta, x_c, sigma, basis, x_t):
    if basis == 'gauss':
        K = np.exp(-sqdist(x_t, x_c)/(2*sigma**2))
    elif basis == 'lm':
        K = np.c_[x_t, np.ones(x_t.shape[0])]
    return K.dot(theta)
