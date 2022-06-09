"""
BERT-base L-12 (Google, UER)
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
        """
        fine tune last 3 layer
        """
        # activate_layer = ["bert.encoder.layer.9.attention.self.query.weight",
        #                   "bert.encoder.layer.9.attention.self.query.bias",
        #                   "bert.encoder.layer.9.attention.self.key.weight",
        #                   "bert.encoder.layer.9.attention.self.key.bias",
        #                   "bert.encoder.layer.9.attention.self.value.weight",
        #                   "bert.encoder.layer.9.attention.self.value.bias",
        #                   "bert.encoder.layer.9.attention.output.dense.weight",
        #                   "bert.encoder.layer.9.attention.output.dense.bias",
        #                   "bert.encoder.layer.9.attention.output.LayerNorm.weight",
        #                   "bert.encoder.layer.9.attention.output.LayerNorm.bias",
        #                   "bert.encoder.layer.9.intermediate.dense.weight",
        #                   "bert.encoder.layer.9.intermediate.dense.bias",
        #                   "bert.encoder.layer.9.output.dense.weight",
        #                   "bert.encoder.layer.9.output.dense.bias",
        #                   "bert.encoder.layer.9.output.LayerNorm.weight",
        #                   "bert.encoder.layer.9.output.LayerNorm.bias",
        #                   "bert.encoder.layer.10.attention.self.query.weight",
        #                   "bert.encoder.layer.10.attention.self.query.bias",
        #                   "bert.encoder.layer.10.attention.self.key.weight",
        #                   "bert.encoder.layer.10.attention.self.key.bias",
        #                   "bert.encoder.layer.10.attention.self.value.weight",
        #                   "bert.encoder.layer.10.attention.self.value.bias",
        #                   "bert.encoder.layer.10.attention.output.dense.weight",
        #                   "bert.encoder.layer.10.attention.output.dense.bias",
        #                   "bert.encoder.layer.10.attention.output.LayerNorm.weight",
        #                   "bert.encoder.layer.10.attention.output.LayerNorm.bias",
        #                   "bert.encoder.layer.10.intermediate.dense.weight",
        #                   "bert.encoder.layer.10.intermediate.dense.bias",
        #                   "bert.encoder.layer.10.output.dense.weight",
        #                   "bert.encoder.layer.10.output.dense.bias",
        #                   "bert.encoder.layer.10.output.LayerNorm.weight",
        #                   "bert.encoder.layer.10.output.LayerNorm.bias",
        #                   "bert.encoder.layer.11.attention.self.query.weight",
        #                   "bert.encoder.layer.11.attention.self.query.bias",
        #                   "bert.encoder.layer.11.attention.self.key.weight",
        #                   "bert.encoder.layer.11.attention.self.key.bias",
        #                   "bert.encoder.layer.11.attention.self.value.weight",
        #                   "bert.encoder.layer.11.attention.self.value.bias",
        #                   "bert.encoder.layer.11.attention.output.dense.weight",
        #                   "bert.encoder.layer.11.attention.output.dense.bias",
        #                   "bert.encoder.layer.11.attention.output.LayerNorm.weight",
        #                   "bert.encoder.layer.11.attention.output.LayerNorm.bias",
        #                   "bert.encoder.layer.11.intermediate.dense.weight",
        #                   "bert.encoder.layer.11.intermediate.dense.bias",
        #                   "bert.encoder.layer.11.output.dense.weight",
        #                   "bert.encoder.layer.11.output.dense.bias",
        #                   "bert.encoder.layer.11.output.LayerNorm.weight",
        #                   "bert.encoder.layer.11.output.LayerNorm.bias",
        #                   "bert.pooler.dense.weight",
        #                   "bert.pooler.dense.bias"]
        # for name, p in self.named_parameters():
        #     if name not in activate_layer:
        #         p.requires_grad = False

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
    # bert_path = r'D:/BERT/pytorch_chinese_L-12_H-768_A-12/'
    # model = BertBiLSTMSoftAtt(bert_path=bert_path,
    #                           hidden_size=128,
    #                           dropout=0.3,
    #                           num_classes=34)
    # print(model)
    pass
