import chainer
import chainer.links as L
import chainer.functions as F
from chainer.datasets import LabeledImageDataset


class CNN_1(chainer.Chain):
    def __init__(self, n_mid=100, n_out=2):
        super().__init__()
        with self.init_scope():
            self.cnv1 = L.Convolution2D(in_channels=3, 
                                        out_channels=16, 
                                        ksize=3,
                                        stride=1,
                                        pad=1)
            self.fc1 = L.Linear(None, n_mid)
            self.fco = L.Linear(None, n_out)
            
    def __call__(self, x):
        h = self.cnv1(x)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 3, 3)
        h = self.fc1(h)
        h = self.fco(h)
        return h