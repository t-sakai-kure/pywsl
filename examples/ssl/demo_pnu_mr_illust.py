import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2.0, rc={"lines.linewidth": 4.0})
sns.set_style('ticks')
from sklearn.model_selection import GridSearchCV

from pywsl.ssl.pnu_mr import PNU_SL
from pywsl.cpe.cpe_ene import cpe
from pywsl.utils.syndata import gen_twonorm_ssl
from pywsl.utils.comcalc import bin_clf_err


if __name__ == "__main__":
    np.random.seed(1)

    n_l = 20
    n_u = 1000
    n_t = 1000

    prior_l = .5
    prior_u = .3

    eta_list = np.arange(-.9, 1, .1)

    n_trial = 20

    best_err = np.inf
    errs1 = np.empty(n_trial)
    errs2 = np.empty((n_trial, len(eta_list)))
    priors = np.empty(n_trial)
    for ite in range(n_trial):
        x, y, x_t, y_t = gen_twonorm_ssl(n_l, prior_l, n_u, prior_u, n_t)
        x_l, y_l, x_u = x[y != 0, :], y[y != 0], x[y == 0, :]
        priorh = cpe(x_l, y_l, x_u)
        clf = PNU_SL(prior=priorh, lam=.1, basis='lm')
        params = {'eta': eta_list}
        clf = GridSearchCV(estimator=clf, param_grid=params, cv=2) #, n_jobs=-1)
        clf.fit(x, y)
        y_h = clf.predict(x_t)
        errs1[ite] = 100*bin_clf_err(y_h, y_t, prior=prior_u)
        if errs1[ite] < best_err:
            best_err = errs1[ite]
            best_w = clf.best_estimator_.coef_
            best_x, best_y = x, y

        for ite_eta, eta in enumerate(eta_list):
            clf = PNU_SL(prior=priorh, eta=eta, lam=.1, basis='lm')
            clf.fit(x, y)
            y_h = clf.predict(x_t)
            errs2[ite, ite_eta] = 100*bin_clf_err(y_h, y_t, prior=prior_u)
                                      
        priors[ite] = priorh

    print("Average of misclassification rates: {:.1f} ({:.2f})".format(
        np.mean(errs1), np.std(errs1)/np.sqrt(n_trial)))
    print("Average of estimated class-priors: {:.2f} ({:.2f})".format(
        np.mean(priors), np.std(priors)/np.sqrt(n_trial)))


    x_p = best_x[best_y == +1, :]
    x_n = best_x[best_y == -1, :]
    x_u = best_x[best_y ==  0, :]


    fig1 = plt.figure(1)
    plt.scatter(x_u[:, 0], x_u[:, 1], c='k', marker='.')
    plt.scatter(x_p[:, 0], x_p[:, 1], facecolors='none', edgecolor='b',
                marker='o', s=300, lw=5)
    plt.scatter(x_n[:, 0], x_n[:, 1], c='r',
                marker='x', s=300, lw=5)

    u1, u2 = np.min(x[:, 0]), np.max(x[:, 0])
    v1_opt = np.log(prior_u/(1-prior_u))/2 - u1
    v2_opt = np.log(prior_u/(1-prior_u))/2 - u2
    plt.plot([u1, u2], [v1_opt, v2_opt])

    w = best_w[:2]
    intercept = best_w[2]
    v1_est = (intercept - w[0]*u1)/w[1]
    v2_est = (intercept - w[0]*u2)/w[1]
    plt.plot([u1, u2], [v1_est, v2_est])
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    sns.despine()
    plt.savefig('data.pdf')
    
    ###
    fig2 = plt.figure(2)
    plt.errorbar(eta_list, np.mean(errs2, axis=0),
                 yerr=np.std(errs2, axis=0)/np.sqrt(n_trial))
    sns.despine()
    plt.savefig('err_curve.pdf')

    plt.show()

