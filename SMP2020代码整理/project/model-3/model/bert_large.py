"""
BERT-large L-24
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
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, input_ids, attention_mask):
        # bert
        encoded_outputs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        encoded_outputs = encoded_outputs[:, 0, :]  # [CLS]

        # dropout
        output = self.dropout(encoded_outputs)

        logits = self.fc(output)

        return logits


if __name__ == "__main__":
    # bert_path = r'D:/BERT/pytorch_chinese_L-24_H-1024_A-16'
    # model = BertFC(bert_path=bert_path,
    #                dropout=0.3,
    #                num_classes=6)
    # input_ids = torch.rand(2, 10).long()
    # attention_mask = torch.ones(2, 10).long()
    # logits = model(input_ids, attention_mask)
    # print(model)
    pass
