# -*- coding: utf-8 -*-
"""
合并疫情和通用训练，集成
"""
import argparse
import json
import os
import platform

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AdamW

from data_utils import log
from data_utils.utils import UtilForCrossValidation
from model.bert_base import BertFC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')


def train(args):
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    logger = log.get_logger(args.log_folder, to_console=True, to_file=True)
    logger.info(str(args))

    util = UtilForCrossValidation(bert_path=args.bert_path,
                                  max_seq_len=args.max_seq_len,
                                  batch_size=args.batch_size,
                                  label_dict_path=args.label_dict_path)
    for fold in range(args.num_fold):
        logger.info(10 * '*' + '{}/{}'.format(fold + 1, args.num_fold) + '*' * 10)

        # 数据加载
        train_loader = util.loader('train', fold, drop_empty=False)
        dev_loader = util.loader('dev', fold, drop_empty=False)

        model = BertFC(args.bert_path,
                       args.dropout,
                       args.num_classes).to(device)

        # 优化器与损失函数
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        # PyTorch scheduler
        # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=3,
        #                                                          num_training_steps=args.epochs)
        criterion = nn.CrossEntropyLoss()

        best_score = 0
        patience = 0
        training_loss = 0
        step = 0
        for epoch in range(args.epochs):
            logger.info(10 * '*' + "training epoch: {} / {}".format(epoch + 1, args.epochs) + '*' * 10)
            # train mode
            model.train()
            for batch in tqdm(train_loader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, mask, labels = batch

                # forward
                logits = model(input_ids, mask)

                # loss
                loss = criterion(logits, labels)
                training_loss += loss.item()

                loss = loss / args.accumulation_steps
                # backward
                loss.backward()

                # 梯度累加
                if (step + 1) % args.accumulation_steps == 0:
                    # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    # scheduler.step()

                # log
                if (step + 1) % args.print_step == 0:
                    logger.info("loss: {}".format(training_loss / args.print_step))
                    training_loss = 0
                step += 1

            # 评估
            dev_acc, dev_f1 = evaluation(model=model, dev_loader=dev_loader)
            logger.info("Validation Accuracy: {}, MacroF1: {}".format(dev_acc, dev_f1))
            if best_score < dev_f1:
                logger.info("Validation macro f1 Improve from {} to {}".format(best_score, dev_f1))
                torch.save(model.state_dict(), os.path.join(args.model_save_path, args.model_name.format(fold)))
                best_score = dev_f1
                patience = 0
            else:
                logger.info("Validation macro f1 don't improve. Best f1: {}".format(best_score))
                patience += 1
                if patience >= args.patience:
                    logger.info("After {} epochs macro f1 don't improve. Break.".format(args.patience))
                    break


def evaluation(model, dev_loader):
    """
    模型在验证集上的正确率
    :param model:
    :param dev_loader:
    :return:
    """
    true = []
    prediction = []
    # eval mode
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dev_loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, mask, labels = batch
            logits = model(input_ids, mask)
            _, pred = torch.max(logits, 1)

            true.append(labels)
            prediction.append(pred)
    true = torch.cat(true).cpu().numpy()
    prediction = torch.cat(prediction).cpu().numpy()
    return accuracy_score(true, prediction), f1_score(true, prediction, average='macro')


def predict(args):
    # 预测，生成结果
    util = UtilForCrossValidation(bert_path=args.bert_path,
                                  max_seq_len=args.max_seq_len,
                                  batch_size=args.batch_size,
                                  label_dict_path=args.label_dict_path)
    index, loader = util.test_loader(is_virus=args.is_virus)
    convert_dict = {value: key for key, value in util.label_dict.items()}
    prediction = torch.zeros(len(index), args.num_classes)
    model = BertFC(args.bert_path,
                   args.dropout,
                   args.num_classes).to(device)

    for fold in range(args.num_fold):
        model_name = os.path.join(args.model_save_path, args.model_name.format(fold))
        print('loading model {}...'.format(model_name))
        model.load_state_dict(torch.load(model_name))
        model.to(device)

        # eval mode
        model.eval()
        temp = []
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, mask = batch
                logits = model(input_ids, mask)  # (Batch, Num_classes)
                temp.append(logits)

        temp = torch.cat(temp).cpu()
        prediction += temp

    prediction = prediction / args.num_fold
    prediction = torch.argmax(prediction, dim=-1).numpy()
    prediction = list(map(lambda x: convert_dict[x], prediction))
    result = []
    for idx, label in zip(index, prediction):
        result.append({"id": idx, "label": label})
    path = args.output_path.format('virus' if args.is_virus else 'usual')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    if platform.system() == 'Windows':
        bert_path = r'D:/BERT/pytorch_chinese_L-24_H-1024_A-16/'
    else:
        bert_path = '/home/BERT/pytorch_chinese_L-24_H-1024_A-16'
    parser.add_argument('--bert_path', type=str, default=bert_path, help='预训练模型(Pytorch)')
    parser.add_argument('--label_dict_path', type=str, default=r'data/label_dict.json', help='标签转换')
    # parser.add_argument('--data_folder', type=str, default=r'../data')
    parser.add_argument('--is_virus', type=bool, default=True, help='virus(True)/usual(False)')
    parser.add_argument('--max_seq_len', type=int, default=300, help='句子最大长度')
    parser.add_argument('--learning_rate', type=float, default=0.00004, help='学习率')
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累加")
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--num_classes', type=int, default=6, help='类别数量')
    parser.add_argument('--num_fold', type=int, default=5, help='交叉验证')
    parser.add_argument('--epochs', type=int, default=4, help='训练轮次')
    parser.add_argument('--patience', type=int, default=2, help='early stopping')
    parser.add_argument('--log_folder', type=str, default='./log', help='日志文件夹')
    parser.add_argument('--model_save_path', type=str, default='./model', help='模型存放目录')
    parser.add_argument('--print_step', type=int, default=300, help='训练时每X步输出loss')
    parser.add_argument('--model_name', type=str, default='bert_large_fold{}.bin', help='模型名称')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--output_path', type=str, default=r'./{}_result.txt', help='提交结果')

    parser.add_argument('--gpu_devices', type=str, default='', help='gpu设备')
    parser.add_argument('--do_train', action='store_true', help='do training procedure?')
    args = parser.parse_args()

    if args.gpu_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

    if args.do_train:
        train(args)

    # virus
    args.is_virus = True
    predict(args)

    # usual
    args.is_virus = False
    predict(args)
