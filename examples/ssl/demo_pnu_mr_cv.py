import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2.0, rc={"lines.linewidth": 4.0})
sns.set_style('ticks')
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from pywsl.ssl.pnu_mr import PNU_SL, PNU_SL_FastCV, pnu_risk, calc_etab
from pywsl.cpe.cpe_ene import cpe
from pywsl.utils.syndata import gen_twonorm_ssl
from pywsl.utils.comcalc import bin_clf_err
from pywsl.utils.timer import Timer


if __name__ == "__main__":
    np.random.seed(1)

    timer = Timer()

    n_l = 20
    n_u = 1000
    n_t = 1000

    prior_l = .5
    prior_u = .3

    eta_list = np.mgrid[-.9:.9:19j]

    x, y, x_t, y_t = gen_twonorm_ssl(n_l, prior_l, n_u, prior_u, n_t)
    x_l, y_l, x_u = x[y != 0, :], y[y != 0], x[y == 0, :]
    priorh = cpe(x_l, y_l, x_u)
    clf = PNU_SL(prior=priorh, basis='lm')
    params = {'eta': eta_list, 'lam': [.1]}
    etah = calc_etab(np.sum(y == +1), np.sum(y == -1), priorh)
    clf = GridSearchCV(estimator=clf, param_grid=params, 
                       scoring=make_scorer(pnu_risk, prior=priorh, eta=etah),
                       cv=3, n_jobs=-1)
    timer.tic("Start PNU_SL")
    clf.fit(x, y)
    timer.toc()
    y_h = clf.predict(x_t)
    err1 = 100*bin_clf_err(y_h, y_t, prior=prior_u)
    print("Error: {:.2f}\n".format(err1))

    timer.tic("Start PNU_SL_FastCV")
    clf2 = PNU_SL_FastCV(x, y, priorh, eta_list, lambda_list=[.1], 
                         n_fold=3, basis='lm', nargout=1)
    timer.toc()
    y_h = clf2(x_t)
    err2 = 100*bin_clf_err(y_h, y_t, prior=prior_u)
    print("Error: {:.2f}".format(err2))

