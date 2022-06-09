# -*- coding: utf-8 -*-
"""
对通用数据集和疫情数据集分别切分5折
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

SEED = 42  # random seed
NUM_FOLD = 5
INPUT_FOLDER = r'../raw'
OUTPUT_FOLDER = r'../data'


if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
if not os.path.exists(os.path.join(OUTPUT_FOLDER, 'train')):
    os.mkdir(os.path.join(OUTPUT_FOLDER, 'train'))
if not os.path.exists(os.path.join(OUTPUT_FOLDER, 'dev')):
    os.mkdir(os.path.join(OUTPUT_FOLDER, 'dev'))

# load data
with open(os.path.join(INPUT_FOLDER, r'train/usual_train.txt'), 'r', encoding='utf-8') as f:
    usual_data = json.load(f)
usual_content = [i['content'] for i in usual_data]
usual_content = np.array(usual_content)
usual_labels = [i['label'] for i in usual_data]
usual_labels = np.array(usual_labels)
with open(os.path.join(INPUT_FOLDER, r'train/virus_train.txt'), 'r', encoding='utf-8') as f:
    virus_data = json.load(f)
virus_content = [i['content'] for i in virus_data]
virus_content = np.array(virus_content)
virus_labels = [i['label'] for i in virus_data]
virus_labels = np.array(virus_labels)
print('Usual data length', len(usual_data),
      '\nvirus data length', len(virus_data))

# split data(no test set)
kfold = StratifiedKFold(n_splits=NUM_FOLD, shuffle=True, random_state=SEED)
# usual data
for index, (train_index, dev_index) in enumerate(kfold.split(usual_content, usual_labels)):
    print('{}/{}'.format(index+1, NUM_FOLD))
    # train
    train_X = usual_content[train_index]
    train_Y = usual_labels[train_index]
    df = pd.DataFrame()
    df['content'] = train_X
    df['labels'] = train_Y
    df.to_csv(os.path.join(OUTPUT_FOLDER, 'train/usual{}.csv'.format(index)), index=0, encoding='utf-8')

    # dev
    dev_X = usual_content[dev_index]
    dev_Y = usual_labels[dev_index]
    df = pd.DataFrame()
    df['content'] = dev_X
    df['labels'] = dev_Y
    df.to_csv(os.path.join(OUTPUT_FOLDER, 'dev/usual{}.csv'.format(index)), index=0, encoding='utf-8')

# virus data
for index, (train_index, dev_index) in enumerate(kfold.split(virus_content, virus_labels)):
    print('{}/{}'.format(index+1, NUM_FOLD))
    # train
    train_X = virus_content[train_index]
    train_Y = virus_labels[train_index]
    df = pd.DataFrame()
    df['content'] = train_X
    df['labels'] = train_Y
    df.to_csv(os.path.join(OUTPUT_FOLDER, 'train/virus{}.csv'.format(index)), index=0, encoding='utf-8')

    # dev
    dev_X = virus_content[dev_index]
    dev_Y = virus_labels[dev_index]
    df = pd.DataFrame()
    df['content'] = dev_X
    df['labels'] = dev_Y
    df.to_csv(os.path.join(OUTPUT_FOLDER, 'dev/virus{}.csv'.format(index)), index=0, encoding='utf-8')

print('Done.')
