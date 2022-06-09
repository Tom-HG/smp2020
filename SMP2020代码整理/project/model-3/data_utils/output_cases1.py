# -*- coding: utf-8 -*-
"""
输出case，单模
"""
import torch
import os
import tqdm
import argparse
import numpy as np
import pandas as pd
from data_utils.preprocess import process
from data_utils.utils import Util
from model import bert_base, bert_large, bilstm
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loader_and_model(args):
    """
    数据加载和模型
    :param args:
    :return:
    """
    # 合并通用和疫情训练集，单模
    util = Util(bert_path=args.bert_path,
                max_seq_len=args.max_seq_len,
                batch_size=args.batch_size,
                label_dict_path=args.label_dict_path)
    loader = util.loader('dev', drop_empty=False)
    model = bert_base.BertFC(args.bert_path,
                             dropout=0.0,
                             num_classes=args.num_classes).to(device)
    # model = bert_large.BertFC(args.bert_path,
    #                              dropout=0.0,
    #                              num_classes=args.num_classes).to(device)

    print('Loading', args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(device)
    model.eval()
    return loader, model


def to_csv(content, processed_content, true, predict, type_, filename):
    df = pd.DataFrame()
    df['content'] = content
    if processed_content:
        df['processed'] = processed_content
    df['true'] = true
    df['predict'] = predict
    if type_:
        df['type'] = type_
    df.to_csv(filename, index=0, encoding='utf-8')


def output(args):
    # read data
    print('Loading {}...'.format(args.input_path))
    df = pd.read_csv(args.input_path, encoding='utf-8')
    content = list(df['content'].astype(str))
    labels = list(df['labels'].astype(str))
    type_ = list(df['type'].astype(str))

    # pre-process
    processed_content = process(content, labels, drop_empty=False)

    # get data_loader and model
    loader, model = loader_and_model(args)

    predict = []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, mask = batch[0], batch[1]
            logits = model(input_ids, mask)
            _, pred = torch.max(logits, 1)
            predict.append(pred)

    predict = torch.cat(predict).cpu().numpy()

    labels = np.array(labels)
    print("Acc: {}, Macro F1: {}".format(accuracy_score(labels, predict),
                                         f1_score(labels, predict, average='macro')))

    # output to file
    to_csv(content=content,
           processed_content=processed_content,
           true=labels,
           predict=predict,
           type_=type_,
           filename=args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    bert_path = r'D:/BERT/pytorch_chinese_L-12_H-768_A-12/'
    parser.add_argument('--bert_path', type=str, default=bert_path, help='预训练模型(Pytorch)')
    parser.add_argument('--is_virus', type=bool, default=True, help='virus(True)/usual(False)')
    parser.add_argument('--max_seq_len', type=int, default=300, help='句子最大长度')
    parser.add_argument('--batch_size', type=int, default=6, help='批次大小')
    parser.add_argument('--num_classes', type=int, default=6, help='类别数量')
    parser.add_argument('--label_dict_path', type=str, default=r'../data/label_dict.json')
    parser.add_argument('--input_path', type=str, required=True, help='验证集文件路径, ../data/dev.csv')
    parser.add_argument('--output_path', type=str, required=True, help='输出文件路径./dev.csv')
    parser.add_argument('--model_save_path', type=str, required=True, help='模型存放目录./model/bert_fc.bin')
    args = parser.parse_args()

    output(args)
