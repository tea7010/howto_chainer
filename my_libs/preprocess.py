import numpy as np
import chainer
from chainer.datasets import LabeledImageDataset
from chainer.datasets import TransformDataset
from chainercv.transforms import resize

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

class Processing_5:
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
        img = img/255
        img = resize(img, (224, 224))

        if np.random.rand() >= 0.5:
            img = img[:, :, ::-1]
        return img

class Processing_6:
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
        img = img/255
        img = resize(img, (100, 100))

        if np.random.rand() >= 0.5:
            img = img[:, :, ::-1]
        return img

class Processing_7:
    def __init__(self):
        pass
    
    def transform(self, x):
        dataset = LabeledImageDataset(x)
        def normarize(in_data):
            img, label = in_data
            img = img/.255
            img = resize(img, (224, 224))
            img = chainer.links.model.vision.vgg.prepare(img)
            return img, label
        
        return TransformDataset(dataset, normarize)

class Processing_8:
    def __init__(self):
        pass
    
    def transform(self, x):
        dataset = LabeledImageDataset(x)
        def normarize(in_data):
            img, label = in_data
            img = img/255
            img = resize(img, (224, 224))
            img = chainer.links.model.vision.vgg.prepare(img)
            return img, label
        
        return TransformDataset(dataset, normarize)

class Processing_9:
    def __init__(self):
        pass
    
    def transform(self, x):
        dataset = LabeledImageDataset(x)
        def normarize(in_data):
            img, label = in_data
            img = img/255
            img = resize(img, (224, 224))
            return img, label
        
        return TransformDataset(dataset, normarize)

class Processing_10:
    def __init__(self):
        pass
    
    def transform(self, x):
        dataset = LabeledImageDataset(x)
        def normarize(in_data):
            img, label = in_data
            img = chainer.links.model.vision.vgg.prepare(img)
            return img, label
        
        return TransformDataset(dataset, normarize)

