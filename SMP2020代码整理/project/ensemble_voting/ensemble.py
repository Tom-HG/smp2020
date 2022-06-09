"""
集成：
1、vote: 简单投票
2、average: 平均
"""
import os
import json
import numpy as np


def voting(predict_labels: np.ndarray, num_classes, output_path, id2labels: dict):
    np.set_printoptions(threshold=np.inf)
    num_samples = predict_labels.shape[1]

    if len(predict_labels.shape) == 3:
        predict_labels = np.argmax(predict_labels, axis=-1)
    assert len(predict_labels.shape) == 2

    res = np.zeros((num_samples, num_classes), dtype=np.int)
    for model in predict_labels:
        for index, label in enumerate(model):
            res[index, label] += 1
    print(res)
    res = np.argmax(res, axis=1).tolist()
    print(res)

    if output_path:
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        output = []
        for i in range(num_samples):
            output.append({"id": i+1, "label": id2labels[res[i]]})
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f)


def average(logits_list, num_classes):
    result = np.zeros((len(logits_list[0]), num_classes))
    for i in logits_list:
        result += i
    result = result / len(logits_list)
    result = np.argmax(result, axis=1)
    return result.tolist()


if __name__ == '__main__':

    label_dict_path = r'../data/label_dict.json'
    files_path = [
        # r'./usual_result-0.txt',
        # r'./usual_result-1.txt',
        # r'./usual_result-2.txt',
        # r'./usual_result-3.txt',
        # r'./usual_result-4.txt',

        r'./virus_result-3.txt',
        r'./virus_result-5.txt',
        r'./virus_result-6.txt',
    ]
    with open(label_dict_path, 'r', encoding='utf-8') as f:
        label_dict = json.load(f)
    predicts = []
    for p in files_path:
        with open(p, 'r', encoding='utf-8') as f:
            temp = json.load(f)
        temp = [label_dict[i['label']] for i in temp]
        predicts.append(temp)
    convert_dict = {value: key for key, value in label_dict.items()}
    voting(np.array(predicts, dtype=np.int),
           num_classes=len(label_dict),
           output_path=r'./virus_result-old.txt',
           id2labels=convert_dict)

