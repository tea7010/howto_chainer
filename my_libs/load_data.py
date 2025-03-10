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

def load_new_dataset(data_dir):
    train_dogs = glob(os.path.join(data_dir, 'train/dog/*.jpg'))
    train_cats = glob(os.path.join(data_dir, 'train/cat/*.jpg'))
    valid_dogs = glob(os.path.join(data_dir, 'validation/dog/*.jpg'))
    valid_cats = glob(os.path.join(data_dir, 'validation/cat/*.jpg'))
    test_dogs = glob(os.path.join(data_dir, 'test_v2/dog/*.jpg'))
    test_cats = glob(os.path.join(data_dir, 'test_v2/cat/*.jpg'))

    # 教師ラベルの作成
    train_label = {
        'cat': 1,
        'dog': 0}

    df = pd.DataFrame({
        'file_path': train_cats + train_dogs + valid_dogs + valid_cats + test_dogs + test_cats,
    })
    df['label'] = df['file_path'].str.split('/', expand=True)[5]
    df['dataset'] = df['file_path'].str.split('/', expand=True)[4]
    df['target'] = df['label'].replace(train_label)
    
    # データセットを作成
    train_df = df[df['dataset'] == 'train']
    valid_df = df[df['dataset'] == 'validation']
    test_df = df[df['dataset'] == 'test_v2']

    train = TupleDataset(train_df['file_path'].values, train_df['target'].values.astype('int32'))
    valid = TupleDataset(valid_df['file_path'].values, valid_df['target'].values.astype('int32'))
    test = TupleDataset(test_df['file_path'].values, test_df['target'].values.astype('int32'))

    return train, valid, test