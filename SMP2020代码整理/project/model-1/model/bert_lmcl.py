"""
BERT-large L-24
"""
import torch.nn as nn
from transformers import BertModel

from losses.lmcl import LMCL_loss


class BertFC(nn.Module):

    def __init__(self, bert_path, dropout, num_classes):
        super(BertFC, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.lmcl = LMCL_loss(self.fc)

    def forward(self, input_ids, attention_mask, lmcl=False, label=None):
        # bert
        if lmcl:
            return self.forward_lmcl(input_ids, attention_mask, label)
        else:
            encoded_outputs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            encoded_outputs = encoded_outputs[:, 0, :]  # [CLS]

            # dropout
            output = self.dropout(encoded_outputs)

            logits = self.fc(output)

            return logits

    def forward_lmcl(self, input_ids, attention_mask, label):
        # bert
        encoded_outputs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        encoded_outputs = encoded_outputs[:, 0, :]  # [CLS]

        # dropout
        # output = self.dropout(encoded_outputs)

        logits, margin_logits = self.lmcl(encoded_outputs, label)
        # logits = self.fc(output)

        return logits, margin_logits


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
