import chainer
import chainer.links as L
import chainer.functions as F
from chainer.datasets import LabeledImageDataset
from chainercv.links.model.resnet import ResNet50
from chainer.links.caffe import CaffeFunction


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


class CNN_2(chainer.Chain):
    def __init__(self, n_out=2):
        super().__init__()
        with self.init_scope():
            self.cnv1 = L.Convolution2D(in_channels=3, 
                                        out_channels=16, 
                                        ksize=3,
                                        stride=1,
                                        pad=1)
            self.fc1 = L.Linear(None, 128)
            self.fco = L.Linear(None, n_out)
           
    def __call__(self, x):
        h = self.cnv1(x)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 3, 3)
        h = self.fc1(h)
        h = self.fco(h)
        return h


class VGG(chainer.Chain):
    def __init__(self, n_out=2):
        super().__init__()

        with self.init_scope():
            self.base = L.VGG16Layers()
            self.fc8 = L.Linear(None, n_out)

    def __call__(self, x):
        h = self.base(x, layers=['fc7'])['fc7']
        h = self.fc8(h)
        return h

class VGG_2(chainer.Chain):
    def __init__(self, n_out=2):
        super().__init__()

        with self.init_scope():
            self.base = L.VGG16Layers()
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 512)
            self.fc8 = L.Linear(None, n_out)

    def __call__(self, x):
        h = self.base(x, layers=['pool5'])['pool5']
        h = self.fc6(h)
        h = F.relu(h)
        h = F.dropout(h)
        h = self.fc7(h)
        h = F.relu(h)
        h = self.fc8(h)
        return h

class VGG_3(chainer.Chain):
    def __init__(self, n_out=2):
        super().__init__()

        with self.init_scope():
            self.base = L.VGG16Layers()
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 512)
            self.fc8 = L.Linear(None, n_out)

    def __call__(self, x):
        h = self.base(x, layers=['pool5'])['pool5']
        h = self.fc6(h)
        h = F.relu(h)
        h = self.fc7(h)
        h = F.relu(h)
        h = self.fc8(h)
        return h

class Resnet50(chainer.Chain):
    def __init__(self, n_out=2):
        super().__init__()

        with self.init_scope():
            self.base = ResNet50()
            self.fc7 = L.Linear(None, n_out)

    def __call__(self, x):
        h = self.base(x, layers=['fc6'])['fc6']
        h = self.fc7(h)
        return h