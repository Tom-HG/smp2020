import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertLSTMSoft(nn.Module):

    def __init__(self, bert_path, dropout, hidden_size, num_classes):
        super(BertLSTMSoft, self).__init__()
        pass

    def forward(self, input_ids, attention_mask, lengths):
        pass


if __name__ == "__main__":
    pass
