# -*- coding: utf-8 -*-
"""
将切分好的5折疫情+通用数据集合并
"""
import pandas as pd

# train
for i in range(5):
    usual = pd.read_csv(r'../data/train/usual{}.csv'.format(i), encoding='utf-8')
    virus = pd.read_csv(r'../data/train/virus{}.csv'.format(i), encoding='utf-8')

    df = pd.concat([usual, virus], axis=0, ignore_index=True)
    df['type'] = ['usual'] * len(usual) + ['virus'] * len(virus)

    df.to_csv(r'../data/train{}.csv'.format(i), index=0)

# eval
for i in range(5):
    usual = pd.read_csv(r'../data/dev/usual{}.csv'.format(i), encoding='utf-8')
    virus = pd.read_csv(r'../data/dev/virus{}.csv'.format(i), encoding='utf-8')

    df = pd.concat([usual, virus], axis=0, ignore_index=True)
    df['type'] = ['usual'] * len(usual) + ['virus'] * len(virus)

    df.to_csv(r'../data/dev{}.csv'.format(i), index=0)
