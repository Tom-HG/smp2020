# -*- coding: utf-8 -*-
"""
输出case，交叉验证
"""
import torch
import os
import tqdm
import argparse
import json
import numpy as np
import pandas as pd
from utils import UtilForCrossValidation, process
from model.bert import BertFC
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    # 合并通用和疫情训练集，交叉验证
    util = UtilForCrossValidation(bert_path=args.bert_path,
                                  max_seq_len=args.max_seq_len,
                                  batch_size=args.batch_size,
                                  label_dict_path=args.label_dict_path)
    convert_dict = {value: key for key, value in util.label_dict.items()}

    for fold in range(1, args.num_fold):
        # read data
        print('Loading {}...'.format(args.input_path.format(fold)))
        df = pd.read_csv(args.input_path.format(fold), encoding='utf-8')
        content = list(df['content'].astype(str))
        labels = list(df['labels'].astype(str))
        type_ = list(df['type'].astype(str))

        # pre-process
        processed_content, _ = process(content, labels, drop_empty=False)

        # data_loader and model
        loader = util.loader('dev', fold=fold, drop_empty=False)
        model = BertFC(args.bert_path,
                       dropout=0.0,
                       num_classes=args.num_classes).to(device)
        # model = bert_large.BertFC(args.bert_path,
        #                              dropout=0.0,
        #                              num_classes=args.num_classes).to(device)

        print('Loading', args.model_save_path.format(fold))
        model.load_state_dict(torch.load(args.model_save_path.format(fold)))
        model.to(device)
        model.eval()

        predict = []
        with torch.no_grad():
            for batch in tqdm.tqdm(loader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, mask = batch[0], batch[1]
                logits = model(input_ids, mask)
                _, pred = torch.max(logits, 1)
                predict.append(pred)

        predict = torch.cat(predict).cpu().numpy()
        predict = list(map(lambda x: convert_dict[x], predict))

        print("Fold:{}, Acc: {}, Macro F1: {}".format(fold, accuracy_score(labels, predict),
                                                      f1_score(labels, predict, average='macro')))

        # output to file
        to_csv(content=content,
               processed_content=processed_content,
               true=labels,
               predict=predict,
               type_=type_,
               filename=args.output_path.format(fold))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    bert_path = r'D:/BERT/pytorch_chinese_L-24_H-1024_A-16/'
    parser.add_argument('--bert_path', type=str, default=bert_path, help='预训练模型(Pytorch)')
    # parser.add_argument('--is_virus', type=bool, default=True, help='virus(True)/usual(False)')
    parser.add_argument('--max_seq_len', type=int, default=150, help='句子最大长度')
    parser.add_argument('--batch_size', type=int, default=3, help='批次大小')
    parser.add_argument('--num_fold', type=int, default=5, help='类别数量')
    parser.add_argument('--num_classes', type=int, default=6, help='类别数量')
    parser.add_argument('--label_dict_path', type=str, default=r'../data/label_dict.json')
    parser.add_argument('--input_path', type=str, default=r'../data/dev{}.csv', help='验证集文件路径')
    parser.add_argument('--output_path', type=str, default=r'./dev_fold{}.csv', help='输出文件路径')
    parser.add_argument('--model_save_path', type=str, default=r'./model/bert_fold{}.bin', help='模型存放目录')
    args = parser.parse_args()

    output(args)

    # ----------------------------------------------------------------------
    # 分开通用和疫情训练集，交叉验证
    # util = UtilForCrossValidationUsualVirus(bert_path=args.bert_path,
    #                                         max_seq_len=args.max_seq_len,
    #                                         batch_size=args.batch_size,
    #                                         is_virus=args.is_virus,
    #                                         label_dict_path=args.label_dict_path)
    # util.loader(data_type='dev', fold=, drop_empty=False)
