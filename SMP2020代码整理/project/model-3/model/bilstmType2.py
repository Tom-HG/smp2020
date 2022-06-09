import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BertLSTMSoft(nn.Module):

    def __init__(self, bert_path, dropout, hidden_size, num_classes):
        super(BertLSTMSoft, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path)

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=1024,
                           hidden_size=hidden_size,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True)
        self.lstm = self.init_lstm_weights(self.lstm)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, input_ids, attention_mask):
        encoded_layers, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # (batch, seqlen, 768), (batch, 768)

        # dropout
        encoded_layers = self.dropout(encoded_layers)

        # pack padded sequence
        # sequence length
        seq_lengths = [mask.tolist().count(1) for mask in attention_mask]
        pack_encoded_layers = pack_padded_sequence(encoded_layers, seq_lengths, batch_first=True, enforce_sorted=False)

        # lstm
        output, (h_last, c_last) = self.lstm(pack_encoded_layers)
        hidden = torch.cat([h_last[0, :, :], h_last[1, :, :]], 1)

        # dropout
        hidden = self.dropout(hidden)

        # decode
        out = self.fc(hidden)

        return out

    def init_lstm_weights(self, LSTM):
        """
        using The Glorot normal initializer, also called Xavier normal initializer to
        initialize the lstm weights
        :param lstm:
        :return:
        """
        for params in LSTM._all_weights:
            for weight in params:
                if "weight" in weight:
                    nn.init.xavier_normal_(getattr(LSTM, weight))
                elif "bias" in weight:
                    nn.init.xavier_normal_(getattr(LSTM, weight).view(-1, 1))
                else:
                    raise RuntimeError("not the weight or bias")
        return LSTM


if __name__ == "__main__":
    model = BertLSTMSoft(
        bert_path="E:/PretrainedModel/chinese_roberta_wwm_large_ext_pytorch",
        dropout=0.5,
        hidden_size=32,
        num_classes=6
    )

    input_ids = torch.ones(2, 300, dtype=torch.long)
    attention_mask = torch.ones(2, 300, dtype=torch.long)
    # print(input_ids.size())
    # print(attention_mask.size())

    out = model(input_ids, attention_mask)
    print(out.size())