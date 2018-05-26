import six
import copy

try:
    from matplotlib import use
    use('Agg')
except ImportError:
    pass

import chainer
from chainer import Variable, functions as F
from chainer import links as L
from chainer.training import extensions
from chainer.datasets import TupleDataset

import numpy as np

from pywsl.utils.models import TLP, MLP
from pywsl.utils.datasets import load_dataset
from pywsl.utils.tools4chainer import pred
from pywsl.pul.chainer.functions.nnpu_risk import PU_Risk
from pywsl.pul.chainer.functions.pu_accuracy import PU_Accuracy


def main():
    gpu, out = -1, "result"
    stepsize = 0.001
    batchsize, epoch = 10000, 10
    beta, gamma = 0., 1.

    data_id, prior = 0, .5
    n_p, n_n, n_u, n_t, n_vp, n_vn, n_vu = 100, 0, 10000, 100, 20, 20, 100
    data_name, x_p, x_n, x_u, y_u, x_t, y_t, x_vp, x_vn, x_vu, y_vu \
        = load_dataset(data_id, n_p, n_n, n_u, prior, n_t, n_vp=n_vp, n_vn=n_vn, n_vu=n_vu)

    x_p, x_n, x_u, x_t, x_vp, x_vn, x_vu = x_p.astype(np.float32), x_n.astype(np.float32), \
        x_u.astype(np.float32), x_t.astype(np.float32), x_vp.astype(np.float32), \
        x_vn.astype(np.float32), x_vu.astype(np.float32), 
    XYtrain = TupleDataset(np.r_[x_p, x_u], np.r_[np.ones(100), np.zeros(10000)].astype(np.int32))
    XYtest = TupleDataset(np.r_[x_vp, x_vu], np.r_[np.ones(20), np.zeros(100)].astype(np.int32))
    train_iter = chainer.iterators.SerialIterator(XYtrain, batchsize)
    test_iter = chainer.iterators.SerialIterator(XYtest, batchsize, repeat=False, shuffle=False)

    loss_type = lambda x: F.sigmoid(-x)
    nnpu_risk = PU_Risk(prior, loss=loss_type, nnPU=True, gamma=gamma, beta=beta)
    pu_acc = PU_Accuracy(prior)

    model = L.Classifier(MLP(), lossfun=nnpu_risk, accfun=pu_acc)
    if gpu >= 0:
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu(gpu)

    optimizer = chainer.optimizers.Adam(alpha=stepsize)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.005))

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = chainer.training.Trainer(updater, (epoch, 'epoch'), out=out)
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PrintReport(
                ['epoch', 'main/loss', 'validation/main/loss',
                 'main/accuracy', 'validation/main/accuracy', 
                 'elapsed_time']))
    key = 'validation/main/accuracy'
    model_name = 'model'
    trainer.extend(extensions.snapshot_object(model, model_name),
                   trigger=chainer.training.triggers.MaxValueTrigger(key))
    if extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name=f'loss_curve.png'))
            trainer.extend(
                extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 
                                      'epoch', file_name=f'accuracy_curve.png'))


    trainer.run()

    yh = pred(model, x_t, batchsize, gpu)
    mr = prior*np.mean(yh[y_t == +1] <= 0) + (1-prior)*np.mean(yh[y_t == -1] >= 0)
    print("mr: {}".format(mr))


if __name__ == "__main__":
    main()
