import numpy as np
import chainer
from chainer.datasets import LabeledImageDataset
from chainer.datasets import TransformDataset

class Processing_1:
    def __init__(self):
        pass
    
    def transform(self, x):
        return LabeledImageDataset(x)


class Processing_2:
    def __init__(self):
        pass

    def transform(self, x):
        dataset = LabeledImageDataset(x)
        def normarize(in_data):
            img, label = in_data
            img = img/.255
            return img, label
        
        return TransformDataset(dataset, normarize)

class Processing_3:
    def __init__(self):
        pass
    
    def transform(self, x):
        dataset = LabeledImageDataset(x)
        def normarize(in_data):
            img, label = in_data
            img = img/.255
            img = chainer.links.model.vision.vgg.prepare(img)
            return img, label
        
        return TransformDataset(dataset, normarize)

class Processing_4:
    def __init__(self):
        pass

    def transform(self, x):
        dataset = LabeledImageDataset(x)
        def augumentaion(in_data):
            img, label = in_data
            img = self._image_process(img)
            return img, label
        
        return TransformDataset(dataset, augumentaion)
    
    def _image_process(self, img):
        img = img/.255
        if np.random.rand() >= 0.5:
            img = img[:, :, ::-1]
        return img
