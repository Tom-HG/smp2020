"""
BERT-large L-24
"""

from transformers.modeling_bert import *


class BertFC(nn.Module):

    def __init__(self, bert_path, dropout, num_classes):
        super(BertFC, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path, output_hidden_states=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # bert
        encoded_outputs, pooled_output, all_hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # encoded_outputs = encoded_outputs[:, 0, :]  # [CLS]

        # dropout
        # output = self.dropout(pooled_output)
        # output = reduce(lambda a, b: a + b, all_hidden_states[1:]) / (len(all_hidden_states) - 1)
        output = torch.cat([hidden_states[:, 0, :].unsqueeze(1) for hidden_states in all_hidden_states[1:]], 1)
        output = output.mean(dim=1)
        logits = self.fc(output)

        return logits


if __name__ == "__main__":
    bert_path = r'pretrain/pytorch_chinese_L-24_H-1024_A-16'
    model = BertFC(bert_path=bert_path,
                   dropout=0.3,
                   num_classes=6)
    input_ids = torch.rand(2, 10).long()
    attention_mask = torch.ones(2, 10).long()
    logits = model(input_ids, attention_mask)
    # print(model)
