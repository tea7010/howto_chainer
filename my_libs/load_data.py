import os
import pandas as pd
from glob import glob
from chainer.datasets import TupleDataset, split_dataset_random

def load_data(data_dir):
    cat_filepaths = glob(os.path.join(data_dir, 'cat/*.jpg'))
    dog_filepaths = glob(os.path.join(data_dir, 'dog/*.jpg'))

    # 教師ラベルの作成
    train_label = {
        'cat': 1,
        'dog': 0}

    df = pd.DataFrame({
        'file_path': cat_filepaths + dog_filepaths})
    df['label'] = df['file_path'].str.split('/', expand=True)[2]
    df['target'] = df['label'].replace(train_label)
    
    # データセットを作成
    dataset = TupleDataset(df['file_path'], df['target'].astype('int32'))

    #train（モデル学習用）, test（モデル評価用）に、７：３にランダムサンプルして分割
    train, test = split_dataset_random(dataset, int(len(dataset)*0.7), seed=0)
    return train, test