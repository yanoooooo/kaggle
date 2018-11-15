import chainer
from chainer import datasets
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
from chainer import training
from chainer import Variable
from chainer.training import extensions
import numpy as np

class Model(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(Model, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units*2)  # n_units -> n_units
            self.l3 = L.Linear(None, n_units)  # n_units -> n_units
            self.l4 = L.Linear(None, n_out)    # n_units -> n_out

    def forward(self, x_data, t_data, train=True):
        x, t = Variable(x_data), Variable(t_data)
        h1 = F.rrelu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        y = self.l4(h3)
        if train is False:
            return F.softmax(y).data
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
