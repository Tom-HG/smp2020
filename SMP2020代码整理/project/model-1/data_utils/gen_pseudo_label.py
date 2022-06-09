#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:rossliang
# create time:2020/7/15 5:21 下午

import argparse
import json
import os


def load_json_to_map(file):
    id2data = dict()
    with open(file, 'r', encoding='utf-8') as f:
        j_data = json.load(f)
    for i in j_data:
        id2data[i['id']] = i
    return id2data


def main(args):
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    label = load_json_to_map(args.result_file)
    for i in data:
        i['label'] = label[i['id']]['label']
    with open(os.path.join(os.path.dirname(args.data_file),
                           'pseudo_' + os.path.basename(args.data_file)), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', help='data raw file')
    parser.add_argument('-r', '--result_file', help='result submit file')
    args = parser.parse_args()
    main(args)
