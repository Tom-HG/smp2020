import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def gen_usual_pseudo():
    # usual pseudo label
    with open(r'../../raw/eval/usual_eval.txt', 'r', encoding='utf-8') as f:
        usual = json.load(f)
    df = pd.DataFrame()
    df['id'] = [i['id'] for i in usual]
    df['content'] = [i['content'] for i in usual]

    files = list(filter(lambda x: 'usual_result' in x, os.listdir(r'./')))
    print(files)
    for index, path in enumerate(files):
        with open(path, 'r', encoding='utf-8') as f:
            temp = json.load(f)
        df['predict{}'.format(index)] = [i['label'] for i in temp]
    # print(df)
    # 选择标签相同的数据
    count = 0
    usual_pseudo = {'content': [], 'label': []}
    for index, row in df.iterrows():
        add = True  # 该条数据是否添加
        for i in range(len(files)):
            if i == 0:
                label = row['predict{}'.format(i)]
            elif row['predict{}'.format(i)] != label:
                add = False
                break
        if add:
            count += 1
            usual_pseudo['content'].append(row['content'])
            usual_pseudo['label'].append(label)

    # for i, j in zip(usual_pseudo['content'], usual_pseudo['label']):
    #     print(i, j)
    print('add {} samples'.format(count))

    # split data
    SEED = 42
    NUM_FOLD = 5
    kfold = StratifiedKFold(n_splits=NUM_FOLD, shuffle=True, random_state=SEED)
    content = np.array(usual_pseudo['content'])
    labels = np.array(usual_pseudo['label'])
    df = pd.DataFrame()
    for index, (train_index, dev_index) in enumerate(kfold.split(content, labels)):
        temp = pd.DataFrame()
        temp['content'] = content[dev_index]
        temp['labels'] = labels[dev_index]
        temp['fold'] = [index] * len(dev_index)

        df = df.append(temp, ignore_index=True)
    # print(df)
    df.to_csv(r'./usual_pseudo.csv', index=0, encoding='utf-8')


def gen_virus_pseudo():
    # virus pseudo label
    with open(r'../../raw/eval/virus_eval.txt', 'r', encoding='utf-8') as f:
        usual = json.load(f)
    df = pd.DataFrame()
    df['id'] = [i['id'] for i in usual]
    df['content'] = [i['content'] for i in usual]

    files = list(filter(lambda x: 'virus_result' in x, os.listdir(r'./')))
    print(files)
    for index, path in enumerate(files):
        with open(path, 'r', encoding='utf-8') as f:
            temp = json.load(f)
        df['predict{}'.format(index)] = [i['label'] for i in temp]
    # print(df)
    # 选择标签相同的数据
    count = 0
    usual_pseudo = {'content': [], 'label': []}
    for index, row in df.iterrows():
        add = True  # 该条数据是否添加
        for i in range(len(files)):
            if i == 0:
                label = row['predict{}'.format(i)]
            elif row['predict{}'.format(i)] != label:
                add = False
                break
        if add:
            count += 1
            usual_pseudo['content'].append(row['content'])
            usual_pseudo['label'].append(label)

    # for i, j in zip(usual_pseudo['content'], usual_pseudo['label']):
    #     print(i, j)
    print('add {} samples'.format(count))

    # split data
    SEED = 42
    NUM_FOLD = 5
    kfold = StratifiedKFold(n_splits=NUM_FOLD, shuffle=True, random_state=SEED)
    content = np.array(usual_pseudo['content'])
    labels = np.array(usual_pseudo['label'])
    df = pd.DataFrame()
    for index, (train_index, dev_index) in enumerate(kfold.split(content, labels)):
        temp = pd.DataFrame()
        temp['content'] = content[dev_index]
        temp['labels'] = labels[dev_index]
        temp['fold'] = [index] * len(dev_index)

        df = df.append(temp, ignore_index=True)
    # print(df)
    df.to_csv(r'./virus_pseudo.csv', index=0, encoding='utf-8')


def merge():
    usual = pd.read_csv(r'./usual_pseudo.csv', encoding='utf-8')
    virus = pd.read_csv(r'./virus_pseudo.csv', encoding='utf-8')

    num_fold = 5
    for fold in range(num_fold):
        df = pd.read_csv(r'../../data/train{}.csv'.format(fold), encoding='utf-8')

        # add usual
        add = usual.loc[usual['fold'] == fold]
        temp = pd.DataFrame()
        temp['content'] = add['content']
        temp['labels'] = add['labels']
        temp['type'] = ['usual'] * len(add)
        df = df.append(temp, ignore_index=True)

        # add virus
        add = virus.loc[virus['fold'] == fold]
        temp = pd.DataFrame()
        temp['content'] = add['content']
        temp['labels'] = add['labels']
        temp['type'] = ['virus'] * len(add)
        df = df.append(temp, ignore_index=True)

        df.to_csv(r'./train{}.csv'.format(fold), index=0, encoding='utf-8')


if __name__ == '__main__':
    # gen_usual_pseudo()
    gen_virus_pseudo()
    # merge()
