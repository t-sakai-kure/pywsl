import six
import copy
import argparse
import chainer

try:
    from matplotlib import use
    use('Agg')
except ImportError:
    pass

from chainer import Variable, functions as F
from chainer import links as L
from chainer.training import extensions
from chainer.datasets import TupleDataset

import numpy as np

from model import TLP, MLP
from pywsl.utils.datasets import load_dataset
from pywsl.pul.chainer.functions.nnpu_risk import PU_Risk
from pywsl.pul.chainer.functions.pu_accuracy import PU_Accuracy


def process_args():
    parser = argparse.ArgumentParser(
        description='non-negative / unbiased PU learning Chainer implementation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', '-b', type=int, default=10000,
                        help='Mini batch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='Zero-origin GPU ID (negative value indicates CPU)')
    parser.add_argument('--preset', '-p', type=str, default=None,
                        choices=['figure1', 'exp-mnist', 'exp-cifar'],
                        help="Preset of configuration\n"+
                             "figure1: The setting of Figure1\n"+
                             "exp-mnist: The setting of MNIST experiment in Experiment\n"+
                             "exp-cifar: The setting of CIFAR10 experiment in Experiment")
    parser.add_argument('--dataset', '-d', default='mnist', type=str, choices=['mnist', 'cifar10'],
                        help='The dataset name')
    parser.add_argument('--n_p', default=100, type=int,
                        help='# of labeled data')
    parser.add_argument('--n_u', default=20000, type=int,
                        help='# of unlabeled data')
    parser.add_argument('--nt_p', default=30, type=int,
                        help='# of labeled data')
    parser.add_argument('--nt_u', default=200, type=int,
                        help='# of unlabeled data')
#    parser.add_argument('--epoch', '-e', default=100, type=int,
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='# of epochs to learn')
    parser.add_argument('--beta', '-B', default=0., type=float,
                        help='Beta parameter of nnPU')
    parser.add_argument('--gamma', '-G', default=1., type=float,
                        help='Gamma parameter of nnPU')
    parser.add_argument('--loss', type=str, default="sigmoid", choices=['logistic', 'sigmoid'],
                        help='The name of a loss function')
    parser.add_argument('--model', '-m', default='3lp', choices=['linear', '3lp', 'mlp'],
                        help='The name of a classification model')
    parser.add_argument('--stepsize', '-s', default=1e-3, type=float,
                        help='Stepsize of gradient method')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()
    if args.gpu >= 0:
        chainer.cuda.check_cuda_available()
        chainer.cuda.get_device_from_id(args.gpu).use()
    if args.preset == "figure1":
        args.n_p = 100
        args.n_u = 59900
        args.nt_p = 50
        args.nt_u = 500
        args.dataset = "mnist"
        args.batchsize = 30000
        args.model = "3lp"
    elif args.preset == "exp-mnist":
        args.labeled = 1000
        args.unlabeled = 60000
        args.dataset = "mnist"
        args.batchsize = 30000
        args.model = "mlp"
    elif args.preset == "exp-cifar":
        args.labeled = 1000
        args.unlabeled = 50000
        args.dataset = "cifar10"
        args.batchsize = 500
        args.model = "cnn"
        args.stepsize = 1e-5
    assert (args.batchsize > 0)
    assert (args.epoch > 0)
    assert (0 < args.n_p < 30000)
    if args.dataset == "mnist":
        assert (0 < args.n_u <= 60000)
    else:
        assert (0 < args.n_u <= 50000)
    assert (0. <= args.beta)
    assert (0. <= args.gamma <= 1.)
    return args


def select_loss(loss_name):
    losses = {"logistic": lambda x: F.softplus(-x), "sigmoid": lambda x: F.sigmoid(-x)}
    return losses[loss_name]


def pred(model, x, batchsize, gpu):
    x = x.astype(np.float32)
    tmp = np.ones(x.shape[0], np.int32)
    it = chainer.iterators.SerialIterator(chainer.datasets.TupleDataset(x, tmp), batchsize,
                                          repeat=False, shuffle=False)
    h = np.empty((0, 1))
    with chainer.configuration.using_config('Train', False):
        with chainer.function.no_backprop_mode():
            for batch in it:
                x, y = chainer.dataset.convert.concat_examples(batch, gpu)
                y = model.predictor(x)
                y.to_cpu()
                h = np.r_[h, y.array]
    return h


def main():
    args = process_args()
    # dataset setup
    prior = .5
#    XYtrain, XYtest = load_dataset(args.dataset, args.n_p, args.n_u, args.nt_p, args.nt_u, prior)
    data_name, x_p, x_n, x_u, y_u, x_t, y_t, x_vp, x_vn, x_vu, y_vu \
        = load_dataset(0, 100, 100, 10000, prior, 100, n_vp=20, n_vn=20, n_vu=100)
    print(x_p.shape, x_u.shape)
#    dim = XYtrain[0][0].size // len(XYtrain[0][0])
    x_p, x_n, x_u, x_t, x_vp, x_vn, x_vu = x_p.astype(np.float32), x_n.astype(np.float32), \
        x_u.astype(np.float32), x_t.astype(np.float32), x_vp.astype(np.float32), \
        x_vn.astype(np.float32), x_vu.astype(np.float32), 
    XYtrain = TupleDataset(np.r_[x_p, x_u], np.r_[np.ones(100), np.zeros(10000)].astype(np.int32))
    XYtest = TupleDataset(np.r_[x_vp, x_vu], np.r_[np.ones(20), np.zeros(100)].astype(np.int32))
    train_iter = chainer.iterators.SerialIterator(XYtrain, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(XYtest, args.batchsize, repeat=False, shuffle=False)
    y_tr = np.array([data[1] for data in XYtrain])
    y_te = np.array([data[1] for data in XYtest])

    print(np.histogram(y_tr))
    print(np.sum(y_tr == 1), np.sum(y_tr == 0))
#    sys.exit(0)

    # model setup
    loss_type = select_loss(args.loss)
    nnpu_loss = PU_Risk(prior, loss=loss_type, nnPU=True, gamma=args.gamma, beta=args.beta)
    pu_acc = PU_Accuracy(prior)

#    model = L.Classifier(TLP(), lossfun=nnpu_loss, accfun=pu_acc)
    model = L.Classifier(MLP(), lossfun=nnpu_loss, accfun=pu_acc)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    # trainer setup
    optimizer = chainer.optimizers.Adam(alpha=args.stepsize)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.005))

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
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

    print("prior: {}".format(prior))
    print("loss: {}".format(args.loss))
    print("batchsize: {}".format(args.batchsize))
    print("beta: {}".format(args.beta))
    print("gamma: {}".format(args.gamma))
    print("")

    # run training
    trainer.run()


#    yh = pred(selected_model, x_t, args.batchsize, args.gpu)
    yh = pred(model, x_t, args.batchsize, args.gpu)
    mr = prior*np.mean(yh[y_t == +1] <= 0) + (1-prior)*np.mean(yh[y_t == -1] >= 0)
    print("mr: {}".format(mr))


if __name__ == "__main__":
    main()
