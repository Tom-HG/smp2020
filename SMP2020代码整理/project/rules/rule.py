# -*- coding: utf-8 -*-
"""
一路走好 -> sad
2、一路走好 + 蜡烛
2、一路走好+[悲伤]

[悲伤]（对疫情处理）
1、[蜡烛]+[悲伤]

[蜡烛]（对疫情处理）
1、纯蜡烛^[蜡烛]$
3、[悲伤] + [蜡烛]
"""
import re
import os
import json
import pandas as pd


def load_eval_data(is_virus: bool, data_folder=r'../raw/test'):
    """
    加载通用和疫情验证集
    :param is_virus: usual or virus
    :param data_folder: 数据文件夹
    :return:
    """
    if is_virus:
        # virus_eval.txt
        path = os.path.join(data_folder, 'virus_test.txt')
    else:
        # usual_eval.txt
        path = os.path.join(data_folder, 'usual_test.txt')

    print('Loading {}...'.format(path))
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame()
    df['id'] = [i['id'] for i in data]
    df['content'] = [i['content'] for i in data]

    return df


def process(predict: list, output_path):
    """
    对疫情验证集修改
    :param predict: 模型输出结果，[{"id": 0, "label": ""}]
    :param output_path: 输出路径
    :return:
    """
    df = load_eval_data(is_virus=True)
    assert len(df) == len(predict)
    predict = [i['label'] for i in predict]

    result = []
    change_count = 0  # 统计有多少数据被修改
    for index, content, predict_label in zip(df['id'].astype(int), df['content'].astype(str), predict):
        if '一路走好' in content:
            result.append({"id": index, "label": "sad"})
            print(index, predict_label, '->', 'sad', content)
            if predict_label != 'sad':
                change_count += 1
        elif '[悲伤]' in content and '[蜡烛]' in content:
            result.append({"id": index, "label": "sad"})
            print(index, predict_label, '->', 'sad', content)
            if predict_label != 'sad':
                change_count += 1
        elif re.match(r'^[\[蜡烛\]]{2,}$', content):
            result.append({"id": index, "label": "sad"})
            print(index, predict_label, '->', 'sad', content)
            if predict_label != 'sad':
                change_count += 1
        else:
            result.append({"id": index, "label": predict_label})
    assert len(result) == len(df)

    # output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f)
    print('Done.Change count:', change_count)


if __name__ == '__main__':
    with open(r'./virus_test.txt', 'r', encoding='utf-8') as f:
        predict_data = json.load(f)
    process(predict=predict_data,
            output_path=r'./virus_temp.txt')
