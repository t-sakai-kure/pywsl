import numpy as np
import chainer


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
                y = chainer.backends.cuda.to_cpu(y.array)
                h = np.r_[h, y]
#                h = np.r_[h, y.array]
    return h
