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
from model import TLP
from dataset import load_dataset

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
    parser.add_argument('--labeled', '-l', default=100, type=int,
                        help='# of labeled data')
    parser.add_argument('--unlabeled', '-u', default=59900, type=int,
                        help='# of unlabeled data')
    parser.add_argument('--epoch', '-e', default=100, type=int,
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
        args.labeled = 100
        args.unlabeled = 59900
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
    assert (0 < args.labeled < 30000)
    if args.dataset == "mnist":
        assert (0 < args.unlabeled <= 60000)
    else:
        assert (0 < args.unlabeled <= 50000)
    assert (0. <= args.beta)
    assert (0. <= args.gamma <= 1.)
    return args


def select_loss(loss_name):
    losses = {"logistic": lambda x: F.softplus(-x), "sigmoid": lambda x: F.sigmoid(-x)}
    return losses[loss_name]


def main():
    args = process_args()
    # dataset setup
    XYtrain, XYtest, prior = load_dataset(args.dataset, args.labeled, args.unlabeled)
    dim = XYtrain[0][0].size // len(XYtrain[0][0])
    train_iter = chainer.iterators.SerialIterator(XYtrain, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(XYtest, args.batchsize, repeat=False, shuffle=False)

    # model setup
    loss_type = select_loss(args.loss)
    nnpu_loss = PU_Risk(prior, loss=loss_type, nnPU=True, gamma=args.gamma, beta=args.beta)
    pu_acc = PU_Accuracy(prior)

#    model = L.Classifier(TLP(), lossfun=nnpu_loss, accfun=pu_acc)
    model = L.Classifier(TLP(), lossfun=nnpu_loss)
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
    if extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name=f'loss_curve.png'))
            trainer.extend(
                extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 
                                      'epoch', file_name=f'accuracy_curve.png'))

    print("prior: {}".format(prior))
    print("loss: {}".format(args.loss))
    print("batchsize: {}".format(args.batchsize))
#    print("model: {}".format(selected_model))
    print("beta: {}".format(args.beta))
    print("gamma: {}".format(args.gamma))
    print("")

    # run training
    trainer.run()


if __name__ == '__main__':
    main()
