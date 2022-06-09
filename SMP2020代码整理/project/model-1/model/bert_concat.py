"""
BERT
RoBERTa
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class BertFC(nn.Module):

    def __init__(self, bert_path, dropout, num_classes, num_layers, add_pooled_output=True):
        """

        :param bert_path:
        :param dropout:
        :param num_classes:
        :param num_layers: 取最后X层的[CLS] hidden state
        :param add_pooled_output: 是否加上pooled_output
        """
        super(BertFC, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.config.output_hidden_states = True
        self.bert = BertModel(self.config)
        self.dropout = nn.Dropout(dropout)

        self.num_layers = num_layers
        self.add_pooled_output = add_pooled_output
        in_features = self.config.hidden_size * num_layers
        if add_pooled_output:
            in_features += self.config.hidden_size

        self.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids:
        :param attention_mask:
        :return:
        """
        '''
        hidden_states (:obj:`tuple(torch.FloatTensor)`,
        `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        embedding layer1 layer2 ... layer12
        '''
        # sequence_output, pooled_output, (hidden_states)
        # (B, T, H) (B, H) (Num_layers, B, T, H)
        sequence_output, pooled_output, hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        outputs = []
        for i in range(-1, -(self.num_layers+1), -1):
            outputs.append(hidden_states[i][:, 0])
        outputs = torch.cat(outputs, dim=-1)  # (B, H*num_layers)
        if self.add_pooled_output:
            # (B, H*num_layers + H)
            outputs = torch.cat([outputs, pooled_output], dim=-1)
        # print(outputs.size())

        # dropout
        outputs = self.dropout(outputs)

        logits = self.fc(outputs)

        return logits


if __name__ == "__main__":
    '''
    D:/BERT/pytorch_chinese_L-12_H-768_A-12
    D:/BERT/pytorch_chinese_L-24_H-1024_A-16
    D:/BERT/pytorch_chinese_weibo_L-12_H-768_A-12
    '''
    # input_ids = torch.rand(2, 5).long()
    # attn_mask = torch.ones(2, 5).long()
    # model = BertFC(bert_path=r'D:\BERT\pytorch_chinese_L-24_H-1024_A-16',
    #                dropout=0.1,
    #                num_classes=6,
    #                num_layers=2,
    #                add_pooled_output=True)
    # logits = model(input_ids, attn_mask)
    # print(logits.size())
    pass
