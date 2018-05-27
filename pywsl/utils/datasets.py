import numpy as np
from sklearn.datasets import fetch_mldata
from chainer.datasets import TupleDataset


def load_dataset(data_id, n_p, n_n, n_u, prior, n_t, n_vp=None, n_vn=None, n_vu=None):
    if data_id == 0:
        data_name = "MNIST"
        x, y = get_mnist()
        x, y = x/255., binarize_mnist(y)
        pos, neg = +1, -1

    data_p, data_n = x[y == pos, :], x[y == neg, :]
    n_up, n_un = split_size(n_u, prior)

    data_p, data_n, x_l, y_l = split_data(data_p, data_n, n_p, n_n)
    data_p, data_n, x_u, y_u = split_data(data_p, data_n, n_up, n_un)
    if n_vp is not None and n_vn is not None:
        data_p, data_n, x_vl, y_vl = split_data(data_p, data_n, n_vp, n_vn)
    if n_vu is not None:
        n_vup, n_vun = split_size(n_vu, prior)
        data_p, data_n, x_vu, y_vu = split_data(data_p, data_n, n_vup, n_vun)
    data_p, data_n, x_t, y_t = split_data(data_p, data_n, n_t, n_t)

    x_p, x_n = x_l[y_l == +1, :], x_l[y_l == -1, :]
    if n_vp is not None and n_vn is not None and n_vu is not None:
        x_vp, x_vn = x_vl[y_vl == +1, :], x_vl[y_vl == -1, :]
        return data_name, x_p, x_n, x_u, y_u, x_t, y_t, x_vp, x_vn, x_vu, y_vu
    if n_vp is not None and n_vn is not None:
        return data_name, x_p, x_n, x_u, y_u, x_t, y_t, x_vp, x_vn
    return data_name, x_p, x_n, x_u, y_u, x_t, y_t



def get_mnist():
    mnist = fetch_mldata('MNIST original', data_home='~')
    x, y = mnist.data, mnist.target
    return x, y

def binarize_mnist(org_y):
    y = np.ones(len(org_y))
    y[org_y % 2 == 1] = -1
    return y

def split_data(data_p, data_n, n_p, n_n):
    N_p, N_n = data_p.shape[0], data_n.shape[0]
    index_p, index_n = split_index(N_p, n_p), split_index(N_n, n_n)
    x_up, x_un = data_p[index_p, :], data_n[index_n, :]
    data_p, data_n = data_p[np.logical_not(index_p), :], data_n[np.logical_not(index_n), :]
    x, y = np.r_[x_up, x_un], np.r_[np.ones(n_p), -np.ones(n_n)]
    return data_p, data_n, x, y

def split_size(n, prior):
    n_p = np.random.binomial(n, prior)
    n_n = n - n_p
    return n_p, n_n
    
def split_index(N, n):
    if n > N:
        raise Exception("""The number of samples is small.
Use large-scale dataset or reduce the size of training data.
""")

    index = np.zeros(N, dtype=bool)
    index[np.random.permutation(N)[:n]] = True
    return index
