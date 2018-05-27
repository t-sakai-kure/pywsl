from chainer.backends import cuda
from chainer import function
from chainer.utils import type_check


class PU_Accuracy(function.Function):

    def __init__(self, prior):
        self.prior = prior
        self.positive = 1
        self.unlabeled = 0
#        self.unlabeled = -1

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype.kind == 'i',
            t_type.shape[0] == x_type.shape[0],
#            t_type.shape == x_type.shape,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y, t = inputs
        y_p, y_u = y[t == self.positive], y[t == self.unlabeled]
        n_p, n_u = xp.maximum(1, y_p.shape[0]), xp.maximum(1, y_u.shape[0])
        f_n, f_p = xp.sum(y_p <= 0)/n_p, xp.sum(y_u >= 0)/n_u
        pu_risk = self.prior*f_n + xp.maximum(0, f_p + self.prior*f_n - self.prior)
        return xp.asarray(1 - pu_risk, dtype=y.dtype),


