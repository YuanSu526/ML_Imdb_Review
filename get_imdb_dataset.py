import os
import sys
import numpy as np
import pandas as pd
import torch, torchtext

def get_IMDB_from_torchtext():
    train_iter = torchtext.datasets.IMDB(split='train')
    test_iter = torchtext.datasets.IMDB(split='test')

    labels, reviews = [], []
    for label, line in train_iter:
        assert label in (1, 2)
        labels.append(label)
        reviews.append(line)
    df_train = pd.DataFrame({'sentiment': labels, 'review': reviews})
    print('original df_train.shape: ', df_train.shape)

    labels, reviews = [], []
    for label, line in test_iter:
        assert label in (1, 2)
        labels.append(label)
        reviews.append(line)
    df_test = pd.DataFrame({'sentiment': labels, 'review': reviews})
    print('original df_test.shape: ', df_test.shape)

    return df_train, df_test

df_train, df_test = get_IMDB_from_torchtext()
df_train.to_csv('./data/imdb_train.csv', index=False)
df_test.to_csv('./data/imdb_test.csv', index=False)

print(df_train.shape, df_test.shape)