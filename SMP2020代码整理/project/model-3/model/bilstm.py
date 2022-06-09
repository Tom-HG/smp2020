"""
Bi-LSTM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformers import BertModel


class BiLSTM(nn.Module):

    def __init__(self, bert_path, dropout, hidden_size, num_classes):
        super(BiLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path)

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(input_size=768,
                           hidden_size=hidden_size,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, input_ids, attention_mask):
        # bert
        embedding_sequence, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embedding_sequence = embedding_sequence[:, 1:, :]  # 去掉[CLS]

        # dropout
        embedding_sequence = self.dropout(embedding_sequence)

        lengths = [mask.tolist().count(1) - 2 for mask in attention_mask]  # 减去[CLS][SEP]
        lengths = torch.tensor(lengths).long()
        sorted_lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        embedding_sequence = embedding_sequence.index_select(0, idx_sort.to(embedding_sequence.device))
        packed_sequence = pack_padded_sequence(embedding_sequence, sorted_lengths, batch_first=True)

        # LSTM
        packed_sequence, _ = self.rnn(packed_sequence)
        padded_sequence, lengths = pad_packed_sequence(packed_sequence, batch_first=True)

        # get real context vector
        outputs = padded_sequence[list(range(padded_sequence.size(0))), lengths - 1, :]

        outputs = self.dropout(outputs)
        logits = self.fc(outputs)

        return logits


if __name__ == "__main__":
    # inputs = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7], [8, 9, 0, 0], [10, 11, 12, 0]]).long()
    # lengths = torch.tensor([3, 4, 2, 3]).long()
    # embed = torch.nn.Embedding(13, 3)
    #
    # sorted_lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
    # inputs = embed(inputs)
    # # print(inputs)
    # inputs = inputs.index_select(0, idx_sort)
    # # print(inputs)
    # inputs = pack_padded_sequence(inputs, sorted_lengths, batch_first=True)
    # # print(inputs)
    # lstm = nn.LSTM(input_size=3, hidden_size=3, batch_first=True, num_layers=1, bidirectional=True)
    # outputs, _ = lstm(inputs)
    # print(outputs)
    # outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
    # print(outputs, outputs.size())
    # print(outputs[list(range(outputs.size(0))), lengths-1, :])
    pass
