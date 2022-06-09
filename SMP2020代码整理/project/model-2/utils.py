# -*- coding: utf-8 -*-
"""
数据加载工具类
"""
import tqdm
import json
import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from preprocess import process


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
        path = r'./data/{}{}.csv'.format(data_type, fold)
        print('Loading {}...'.format(path))
        # content,labels,type
        df = pd.read_csv(path, encoding='utf-8')
        content = list(df['content'].astype(str))
        labels = list(df['labels'].astype(str))
        type_ = list(df['type'].astype(str))  # usual or virus

        # 训练集，删除重复的，句子和标签一致
        if data_type == 'train':
            result_content = []
            result_labels = []
            result_type = []
            dataset = {}
            for sentence, label, t in zip(content, labels, type_):
                if sentence in dataset:
                    if dataset[sentence] != label:
                        result_content.append(sentence)
                        result_labels.append(label)
                        result_type.append(t)
                else:
                    result_content.append(sentence)
                    result_labels.append(label)
                    result_type.append(t)
                    dataset[sentence] = label
            return result_content, result_labels, result_type
        else:
            return content, labels, type_

    def loader(self, data_type: str, fold: int, drop_empty=False):
        content, labels, type_ = self.read_data(data_type, fold)
        # 训练-过采样，对疫情的sad fear surprise扩大一倍
        if data_type == 'train':
            over_sampling = {'content': [], 'labels': []}
            for sentence, tag, t in zip(content, labels, type_):
                if t == 'virus' and tag in ['sad', 'fear', 'surprise']:
                    over_sampling['content'].append(sentence)
                    over_sampling['labels'].append(tag)
            content.extend(over_sampling['content'])
            labels.extend(over_sampling['labels'])

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
        path = r'../raw/test/{}_test.txt'.format(type_)
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


if __name__ == "__main__":
    # util = UtilForCrossValidation(bert_path=r'D:/BERT/pytorch_chinese_L-24_H-1024_A-16/',
    #                               max_seq_len=300,
    #                               batch_size=5,
    #                               label_dict_path=r'../data/label_dict.json')
    # loader = util.loader('train', fold=1)
    pass
