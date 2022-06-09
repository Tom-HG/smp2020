# -*- coding: utf-8 -*-
"""
数据加载工具类
"""
import json
import random
from collections import defaultdict

import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer

from .preprocess import process, process_emoji


class Util(object):
    """
    加载训练集、验证集
    train.csv dev.csv
    """

    def __init__(self, bert_path, max_seq_len, batch_size, label_dict_path):
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        with open(label_dict_path, 'r', encoding='utf-8') as f:
            self.label_dict = json.load(f)

    def convert_data_to_tensors(self, data):
        input_ids = []
        attention_mask = []
        for sentence in tqdm.tqdm(data):
            token_sentence = self.tokenizer.encode(sentence,
                                                   add_special_tokens=True,
                                                   max_length=self.max_seq_len)
            pad = [0] * (self.max_seq_len - len(token_sentence))
            mask = [1] * len(token_sentence)
            token_sentence.extend(pad)
            mask.extend(pad)

            input_ids.append(token_sentence)
            attention_mask.append(mask)

        input_ids = torch.tensor(input_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        return input_ids, attention_mask

    def read_data(self, data_type: str):
        """
        :param data_type: train or dev
        :return:
        """
        path = r'data/{}.csv'.format(data_type)
        print('Loading {}...'.format(path))
        # content,labels,type
        df = pd.read_csv(path, encoding='utf-8')
        content = list(df['content'].astype(str))
        labels = list(df['labels'].astype(str))
        type_ = list(df['type'].astype(str))  # usual or virus

        return content, labels, type_

    def loader(self, data_type: str, drop_empty=True):
        content, labels, type_ = self.read_data(data_type)
        content, labels = process(content, labels, drop_empty=drop_empty)

        input_ids, attention_mask = self.convert_data_to_tensors(content)
        labels = list(map(lambda x: self.label_dict[x], labels))
        labels = torch.tensor(labels).long()

        dataset = TensorDataset(input_ids, attention_mask, labels)
        shuffle = True if data_type == 'train' else False
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=shuffle,
                            drop_last=False)
        return loader

    def test_loader(self, is_virus):
        type_ = 'virus' if is_virus else 'usual'
        path = r'raw/eval/{}_eval.txt'.format(type_)
        print('Loading {}...'.format(path))
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        index = [i['id'] for i in data]
        content = [i['content'] for i in data]
        content, _ = process(content, [0] * len(content), drop_empty=False)

        input_ids, attention_mask = self.convert_data_to_tensors(content)
        dataset = TensorDataset(input_ids, attention_mask)
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=16)
        return index, loader


class UtilForCrossValidation(object):
    """
    加载交叉验证的训练集、验证集
    train0~4.csv dev0~4.csv
    """

    def __init__(self, bert_path, max_seq_len, batch_size, label_dict_path):
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        with open(label_dict_path, 'r', encoding='utf-8') as f:
            self.label_dict = json.load(f)

    def convert_data_to_tensors(self, data):
        input_ids = []
        attention_mask = []
        for sentence in tqdm.tqdm(data):
            token_sentence = self.tokenizer.encode(sentence,
                                                   add_special_tokens=True,
                                                   max_length=self.max_seq_len)
            pad = [0] * (self.max_seq_len - len(token_sentence))
            mask = [1] * len(token_sentence)
            token_sentence.extend(pad)
            mask.extend(pad)

            input_ids.append(token_sentence)
            attention_mask.append(mask)

        input_ids = torch.tensor(input_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        return input_ids, attention_mask

    def read_data(self, data_type: str, fold: int):
        """
        :param data_type: train or dev
        :param fold: 0~4
        :return:
        """
        path = r'data/{}{}.csv'.format(data_type, fold)
        print('Loading {}...'.format(path))
        # content,labels,type
        df = pd.read_csv(path, encoding='utf-8')
        content = list(df['content'].astype(str))
        labels = list(df['labels'].astype(str))
        type_ = list(df['type'].astype(str))  # usual or virus

        # 训练集，删除重复的，句子和标签一致
        # if data_type == 'train':
        #     result_content = []
        #     result_labels = []
        #     result_type = []
        #     dataset = {}
        #     for sentence, label, t in zip(content, labels, type_):
        #         if sentence in dataset:
        #             if dataset[sentence] != label:
        #                 result_content.append(sentence)
        #                 result_labels.append(label)
        #                 result_type.append(t)
        #         else:
        #             result_content.append(sentence)
        #             result_labels.append(label)
        #             result_type.append(t)
        #             dataset[sentence] = label
        #     return result_content, result_labels, result_type
        # else:
        #     return content, labels, type_
        return content, labels, type_

    def loader(self, data_type: str, fold: int, drop_empty=False):

        # def random_concat(list1, list2, num):
        #     sent1 = random.sample(list1, num)
        #     sent2 = random.sample(list2, num)
        #     ret = []
        #     for s1, s2 in zip(sent1, sent2):
        #         ret.append(s1 + s2)
        #     return ret

        content, labels, type_ = self.read_data(data_type, fold)
        # # 训练-过采样，对疫情的sad fear surprise扩大一倍
        # if data_type == 'train':
        #     over_sampling = {'content': [], 'labels': []}
        #     for sentence, tag, t in zip(content, labels, type_):
        #         if t == 'virus' and tag in ['sad', 'fear', 'surprise']:
        #             over_sampling['content'].append(sentence)
        #             over_sampling['labels'].append(tag)
        #     content.extend(over_sampling['content'])

        # oversampling bases on concatenation
        # if data_type == 'train':
        #
        #     data = {'virus': defaultdict(list), 'usual': defaultdict(list)}
        #     for sentence, tag, t in zip(content, labels, type_):
        #         data[t][tag].append(sentence)
        #     da_fear = random_concat(data['usual']['fear'] + data['virus']['fear'],
        #                             data['usual']['fear'] + data['virus']['fear'],
        #                             len(data['usual']['fear']) + len(data['virus']['fear']))
        #
        #     da_sad = random_concat(data['usual']['sad'] + data['virus']['sad'],
        #                            data['usual']['sad'] + data['virus']['sad'],
        #                            len(data['usual']['sad']) + len(data['virus']['sad']))
        #
        #     da_usual_surprise = random_concat(data['usual']['surprise'],
        #                                       data['usual']['surprise'],
        #                                       len(data['usual']['surprise']))
        #
        #     da_virus_surprise = random_concat(data['virus']['surprise'],
        #                                       data['virus']['surprise'],
        #                                       len(data['virus']['surprise']))
        #
        #     content.extend(da_fear)
        #     labels.extend(['fear'] * len(da_fear))
        #     content.extend(da_sad)
        #     labels.extend(['sad'] * len(da_sad))
        #     content.extend(da_usual_surprise)
        #     labels.extend(['surprise'] * len(da_usual_surprise))
        #     content.extend(da_virus_surprise)
        #     labels.extend(['surprise'] * len(da_virus_surprise))

        content, labels = process(content, labels, drop_empty=drop_empty)

        input_ids, attention_mask = self.convert_data_to_tensors(content)
        labels = list(map(lambda x: self.label_dict[x], labels))
        labels = torch.tensor(labels).long()

        dataset = TensorDataset(input_ids, attention_mask, labels)
        shuffle = True if data_type == 'train' else False
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=shuffle,
                            drop_last=False)
        return loader

    def test_loader(self, is_virus: bool):
        type_ = 'virus' if is_virus else 'usual'
        path = r'raw/eval/{}_eval.txt'.format(type_)
        print('Loading {}...'.format(path))
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        index = [i['id'] for i in data]
        content = [i['content'] for i in data]
        content, _ = process(content, [0] * len(content), drop_empty=False)

        input_ids, attention_mask = self.convert_data_to_tensors(content)
        dataset = TensorDataset(input_ids, attention_mask)
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            drop_last=False)
        return index, loader


class UtilForCrossValidationUsualVirus(object):
    """
    加载通用数据、疫情数据交叉验证的训练集、验证集
    train/usual0~4.csv train/virus0~4.csv
    dev/usual0~4.csv dev/virus0~4.csv
    """

    def __init__(self, bert_path, max_seq_len, batch_size, is_virus, label_dict_path):
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.type_ = 'virus' if is_virus else 'usual'
        with open(label_dict_path, 'r', encoding='utf-8') as f:
            self.label_dict = json.load(f)

    def convert_data_to_tensors(self, data):
        input_ids = []
        attention_mask = []
        for sentence in tqdm.tqdm(data):
            token_sentence = self.tokenizer.encode(sentence,
                                                   add_special_tokens=True,
                                                   max_length=self.max_seq_len)
            pad = [0] * (self.max_seq_len - len(token_sentence))
            mask = [1] * len(token_sentence)
            token_sentence.extend(pad)
            mask.extend(pad)

            input_ids.append(token_sentence)
            attention_mask.append(mask)

        input_ids = torch.tensor(input_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        return input_ids, attention_mask

    def read_data(self, data_type: str, fold: int):
        """
        :param data_type: train or dev
        :param fold: 0~4
        :return:
        """
        path = r'data/{}/{}{}.csv'.format(data_type, self.type_, fold)
        print('Loading {}...'.format(path))
        # content,labels,type
        df = pd.read_csv(path, encoding='utf-8')
        content = list(df['content'].astype(str))
        labels = list(df['labels'].astype(str))

        return content, labels

    def loader(self, data_type: str, fold: int, drop_empty=False):
        content, labels = self.read_data(data_type, fold)
        content, labels = process(content, labels, drop_empty=drop_empty)

        input_ids, attention_mask = self.convert_data_to_tensors(content)
        labels = list(map(lambda x: self.label_dict[x], labels))
        labels = torch.tensor(labels).long()

        dataset = TensorDataset(input_ids, attention_mask, labels)
        shuffle = True if data_type == 'train' else False
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=shuffle,
                            drop_last=False)
        return loader

    def test_loader(self, is_virus):
        type_ = 'virus' if is_virus else 'usual'
        path = r'raw/eval/{}_eval.txt'.format(type_)
        type_ = 'virus' if is_virus else 'usual'
        path = r'raw/eval/{}_eval.txt'.format(type_)
        print('Loading {}...'.format(path))
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        index = [i['id'] for i in data]
        content = [i['content'] for i in data]
        content, _ = process(content, [0] * len(content), drop_empty=False)

        input_ids, attention_mask = self.convert_data_to_tensors(content)
        dataset = TensorDataset(input_ids, attention_mask)
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            drop_last=False)
        return index, loader


class UtilEmoji(object):
    """
    加载训练集、验证集，处理表情
    train.csv dev.csv
    """

    def __init__(self, vocab_path, max_seq_len, batch_size, label_dict_path, emoji_dict_path):
        # 将99个unused替换成由训练集统计出的出现次数最高的99个表情（统计前图片表情已转文字）
        with open(vocab_path, 'r', encoding='utf-8') as f:
            additional_emoji = [line.strip() for line in f.readlines()]
        additional_emoji = additional_emoji[1: 100]
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path,
                                                       additional_special_tokens=additional_emoji)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        with open(label_dict_path, 'r', encoding='utf-8') as f:
            self.label_dict = json.load(f)
        with open(emoji_dict_path, 'r', encoding='utf-8') as f:
            self.emoji_dict = json.load(f)

    def convert_data_to_tensors(self, data):
        input_ids = []
        attention_mask = []
        for sentence in tqdm.tqdm(data):
            token_sentence = self.tokenizer.encode(sentence,
                                                   add_special_tokens=True,
                                                   max_length=self.max_seq_len)
            # 删除重复的表情，队列，与队尾元素相同则跳过
            temp = []
            for ids in token_sentence:
                if 1 <= ids <= 99:
                    if ids != temp[-1]:
                        temp.append(ids)  # 是表情，且没有连续出现
                else:
                    # 非表情
                    temp.append(ids)
            token_sentence = temp

            pad = [0] * (self.max_seq_len - len(token_sentence))
            mask = [1] * len(token_sentence)
            token_sentence.extend(pad)
            mask.extend(pad)

            input_ids.append(token_sentence)
            attention_mask.append(mask)

        input_ids = torch.tensor(input_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        return input_ids, attention_mask

    def read_data(self, data_type: str):
        """
        :param data_type: train or dev
        :return:
        """
        path = r'data/{}.csv'.format(data_type)
        print('Loading {}...'.format(path))
        # content,labels,type
        df = pd.read_csv(path, encoding='utf-8')
        content = list(df['content'].astype(str))
        labels = list(df['labels'].astype(str))
        type_ = list(df['type'].astype(str))  # usual or virus

        return content, labels, type_

    def loader(self, data_type: str, drop_empty=False):
        content, labels, type_ = self.read_data(data_type)
        content, labels = process(content, labels, drop_empty=drop_empty)
        content = process_emoji(content, self.emoji_dict)

        input_ids, attention_mask = self.convert_data_to_tensors(content)
        labels = list(map(lambda x: self.label_dict[x], labels))
        labels = torch.tensor(labels).long()

        dataset = TensorDataset(input_ids, attention_mask, labels)
        shuffle = True if data_type == 'train' else False
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=shuffle,
                            drop_last=False)
        return loader

    def test_loader(self, is_virus):
        type_ = 'virus' if is_virus else 'usual'
        path = r'raw/eval/{}_eval.txt'.format(type_)
        print('Loading {}...'.format(path))
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        index = [i['id'] for i in data]
        content = [i['content'] for i in data]
        content, _ = process(content, [0] * len(content), drop_empty=False)
        content = process_emoji(content, self.emoji_dict)

        input_ids, attention_mask = self.convert_data_to_tensors(content)
        dataset = TensorDataset(input_ids, attention_mask)
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            drop_last=False)
        return index, loader


class UtilForCrossValidationEmoji(object):
    """
    加载交叉验证的训练集、验证集
    train0~4.csv dev0~4.csv
    """

    def __init__(self, vocab_path, max_seq_len, batch_size, label_dict_path, emoji_dict_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            additional_emoji = [line.strip() for line in f.readlines()]
        additional_emoji = additional_emoji[1: 100]
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path,
                                                       additional_special_tokens=additional_emoji)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        with open(label_dict_path, 'r', encoding='utf-8') as f:
            self.label_dict = json.load(f)

        with open(emoji_dict_path, 'r', encoding='utf-8') as f:
            self.emoji_dict = json.load(f)

    def convert_data_to_tensors(self, data):
        input_ids = []
        attention_mask = []
        for sentence in tqdm.tqdm(data):
            token_sentence = self.tokenizer.encode(sentence,
                                                   add_special_tokens=True,
                                                   max_length=self.max_seq_len)
            # 删除重复的表情，队列，与队尾元素相同则跳过
            temp = []
            for ids in token_sentence:
                if 1 <= ids <= 99:
                    if ids != temp[-1]:
                        temp.append(ids)  # 是表情，且没有连续出现
                else:
                    # 非表情
                    temp.append(ids)
            token_sentence = temp

            pad = [0] * (self.max_seq_len - len(token_sentence))
            mask = [1] * len(token_sentence)
            token_sentence.extend(pad)
            mask.extend(pad)

            input_ids.append(token_sentence)
            attention_mask.append(mask)

        input_ids = torch.tensor(input_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        return input_ids, attention_mask

    def read_data(self, data_type: str, fold: int):
        """
        :param data_type: train or dev
        :param fold: 0~4
        :return:
        """
        path = r'data/{}{}.csv'.format(data_type, fold)
        print('Loading {}...'.format(path))
        # content,labels,type
        df = pd.read_csv(path, encoding='utf-8')
        content = list(df['content'].astype(str))
        labels = list(df['labels'].astype(str))
        type_ = list(df['type'].astype(str))  # usual or virus

        return content, labels, type_

    def loader(self, data_type: str, fold: int, drop_empty=False):
        content, labels, type_ = self.read_data(data_type, fold)
        content, labels = process(content, labels, drop_empty=drop_empty)
        content = process_emoji(content, self.emoji_dict)

        input_ids, attention_mask = self.convert_data_to_tensors(content)
        labels = list(map(lambda x: self.label_dict[x], labels))
        labels = torch.tensor(labels).long()

        dataset = TensorDataset(input_ids, attention_mask, labels)
        shuffle = True if data_type == 'train' else False
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=shuffle,
                            drop_last=False)
        return loader

    def test_loader(self, is_virus: bool):
        type_ = 'virus' if is_virus else 'usual'
        path = r'raw/eval/{}_eval.txt'.format(type_)
        print('Loading {}...'.format(path))
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        index = [i['id'] for i in data]
        content = [i['content'] for i in data]
        content, _ = process(content, [0] * len(content), drop_empty=False)
        content = process_emoji(content, self.emoji_dict)

        input_ids, attention_mask = self.convert_data_to_tensors(content)
        dataset = TensorDataset(input_ids, attention_mask)
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            drop_last=False)
        return index, loader


if __name__ == "__main__":
    # utils = Utils(bert_path=r'D:/BERT/pytorch_chinese_L-12_H-768_A-12/',
    #               is_virus=False, max_seq_len=500, batch_size=5,
    #               label_dict={'happy': 0, 'neural': 1, 'surprise': 2, 'angry': 3, 'sad': 4, 'fear': 5})
    # loader = utils.loader('train')
    pass
