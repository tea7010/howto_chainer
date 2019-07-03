import pandas as pd
import json
import os
from datetime import datetime

import chainer
import chainer.links as L
import chainer.functions as F

from chainer.iterators import SerialIterator
from chainer.optimizers import Adam

from chainer.training import StandardUpdater, Trainer
from chainer.training.extensions import Evaluator, LogReport, PrintReport


class ChainerPipeline:
    def __init__(self, preprocess, network, train, test, setting):
        self.preprocess = preprocess
        self.network = network
        
        self.train = train
        self.test = test
        self.setting = setting
        
    def run(self):
        # 前処理
        train = self.preprocess.transform(self.train)
        test = self.preprocess.transform(self.test)

        # モデル学習・評価
        model = self.chainer_model_pipe(self.network, train, test, self.setting)

        # 結果可視化
        result = self.visualize_result(self.preprocess, self.network, self.setting)
        return model, result
    
    # chainerモデルのパイプライン
    def chainer_model_pipe(self, nn, train, test, params):
        epoch = params['epoch']
        batch_size = params['batch_size']
        use_gpu=params['use_gpu']

        # Model Instance
        model = L.Classifier(nn)

        if use_gpu:
            device = 0
            model.to_gpu(device)
        else:
            device = -1

        # ミニバッチのインスタンスを作成
        train_iter = SerialIterator(train, batch_size)
        test_iter = SerialIterator(test, batch_size, repeat=False, shuffle=False)

        # Set Lerning
        optimizer = Adam()
        optimizer.setup(model)

        updater = StandardUpdater(train_iter, optimizer, device=device)

        trainer = Trainer(updater, (epoch, 'epoch'), out='result/cat_dog')
        trainer.extend(Evaluator(test_iter, model, device=device))
        trainer.extend(LogReport(trigger=(1, 'epoch')))
        trainer.extend(PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy', 
                                    'main/loss', 'validation/main/loss', 'elapsed_time']), 
                       trigger=(1, 'epoch'))

        trainer.run()

        if use_gpu:
            model.to_cpu()

        return model
    
    # 結果の可視化
    def visualize_result(self, preprocess, network, setting, plot_learn=True, path='./result/cat_dog/'):
        with open(os.path.join(path, 'log')) as f:
            result = pd.DataFrame(json.load(f))

        log = pd.Series()
        log['Preprocess'] = preprocess.__class__.__name__
        log['Model'] = network.__class__.__name__
        log['Elapsed time'] = result.iloc[-1]['elapsed_time']
        log['Validation accuracy'] = result.iloc[-1]['validation/main/accuracy']
        log = log.append(pd.Series(setting))
        print(log)
        log.to_json(os.path.join(path, 'run', '%s.json' % datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                  
        if plot_learn:
            result[['main/accuracy', 'validation/main/accuracy']].plot()
        return result