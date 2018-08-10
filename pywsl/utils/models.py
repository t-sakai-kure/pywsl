import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, cuda

class TLP(Chain):
    def __init__(self):
        super(TLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 100)
            self.l2 = L.Linear(None, 1)

        self.af = F.relu

    def __call__(self, x):
        h = self.af(self.l1(x))
        h = self.l2(h)
        return h


class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 100)
            self.b1 = L.BatchNormalization(100)
            self.l2 = L.Linear(None, 100)
            self.b2 = L.BatchNormalization(100)
            self.l3 = L.Linear(None, 100)
            self.b3 = L.BatchNormalization(100)
            self.l4 = L.Linear(None, 1)

        self.af = F.relu


    def __call__(self, x):
        h = self.af(self.b1(self.l1(x)))
        h = self.af(self.b2(self.l2(h)))
        h = self.af(self.b3(self.l3(h)))
        h = self.l4(h)
        return h


class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(3, 96, 3, pad=1),
            conv2=L.Convolution2D(96, 96, 3, pad=1),
            conv3=L.Convolution2D(96, 96, 3, pad=1, stride=2),
            conv4=L.Convolution2D(96, 192, 3, pad=1),
            conv5=L.Convolution2D(192, 192, 3, pad=1),
            conv6=L.Convolution2D(192, 192, 3, pad=1, stride=2),
            conv7=L.Convolution2D(192, 192, 3, pad=1),
            conv8=L.Convolution2D(192, 192, 1),
            conv9=L.Convolution2D(192, 10, 1),
            b1=L.BatchNormalization(96),
            b2=L.BatchNormalization(96),
            b3=L.BatchNormalization(96),
            b4=L.BatchNormalization(192),
            b5=L.BatchNormalization(192),
            b6=L.BatchNormalization(192),
            b7=L.BatchNormalization(192),
            b8=L.BatchNormalization(192),
            b9=L.BatchNormalization(10),
            fc1=L.Linear(None, 1000),
            fc2=L.Linear(1000, 1000),
            fc3=L.Linear(1000, 1),
        )
        self.af = F.relu

    def __call__(self, x):
        h = self.conv1(x)
        h = self.b1(h)
        h = self.af(h)
        h = self.conv2(h)
        h = self.b2(h)
        h = self.af(h)
        h = self.conv3(h)
        h = self.b3(h)
        h = self.af(h)
        h = self.conv4(h)
        h = self.b4(h)
        h = self.af(h)
        h = self.conv5(h)
        h = self.b5(h)
        h = self.af(h)
        h = self.conv6(h)
        h = self.b6(h)
        h = self.af(h)
        h = self.conv7(h)
        h = self.b7(h)
        h = self.af(h)
        h = self.conv8(h)
        h = self.b8(h)
        h = self.af(h)
        h = self.conv9(h)
        h = self.b9(h)
        h = self.af(h)
        h = self.fc1(h)
        h = self.af(h)
        h = self.fc2(h)
        h = self.af(h)
        h = self.fc3(h)
        return h
