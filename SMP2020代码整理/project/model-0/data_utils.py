import re
import numpy as np
import json
from transformers import BertTokenizer
import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from langconv import Converter


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

        return content, labels, type_

    def loader(self, data_type: str, fold: int, drop_empty=False):
        content, labels, type_ = self.read_data(data_type, fold)
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


def process(data, labels, drop_empty=False):
    x = []
    y = []
    for content, label in zip(data, labels):
        content = re.sub(r'(//@.*?:)', '', content)  # 微博用户昵称
        content = re.sub(r'http://[0-9a-zA-Z\./]+', '', content)  # 超链接
        content = re.sub(r'[【】#@:/“”\s·。➕Этойвлшебню\'архчьЯмгсу■\-=●]', '', content)  # 无意义的符号
        content = re.sub(r'(\(ω\)★)|(→_→)|(•̭̆•̆)|(（｡ò∀ó｡）)|(_\(:_」∠\)_)|(↖\(ω\)↗)|(╮⊙o⊙╭╮)|(\(╥ω╥`\))|(&amp;gt;)|(\-_\-\|\|)|\(◍˃̶ᗜ˂̶◍\)ﾉ', '', content)  # 颜文字
        content = re.sub(r'(@王晨MuKii)|(@UNIQ-王一博)|(@Les_etoiles\-肖战)|(@X玖少年团肖战DAYTOY)', '', content)  # @某人
        content = re.sub(r'[0-9\.%]+', 'NUM', content)  # 纯数字
        content = Converter("zh-hans").convert(content)  # 繁体字转简体字
        if len(content) == 0:
            if not drop_empty:
                x.append('空')
                y.append(label)
        else:
            x.append(content)
            y.append(label)
    return x, y


if __name__ == '__main__':
    pass
