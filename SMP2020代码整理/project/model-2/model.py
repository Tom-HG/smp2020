"""
BERT
RoBERTa
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertFC(nn.Module):

    def __init__(self, bert_path, dropout, num_classes):
        super(BertFC, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # bert
        encoded_outputs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        encoded_outputs = encoded_outputs[:, 0, :]  # [CLS]

        # dropout
        output = self.dropout(encoded_outputs)

        logits = self.fc(output)

        return logits


if __name__ == "__main__":
    '''
    D:\BERT\pytorch_chinese_L-12_H-768_A-12
    D:\BERT\pytorch_chinese_L-24_H-1024_A-16
    D:\BERT\pytorch_chinese_weibo_L-12_H-768_A-12
    '''
    # input_ids = torch.rand(5, 20).long()
    # attn_mask = torch.ones(5, 20).long()
    # model = BertFC(bert_path=r'D:\BERT\pytorch_chinese_L-24_H-1024_A-16',
    #                dropout=0.1,
    #                num_classes=6)
    # logits = model(input_ids, attn_mask)
    # print(logits.size())
    pass
