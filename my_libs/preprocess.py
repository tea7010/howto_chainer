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