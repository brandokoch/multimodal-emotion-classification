import torch
import torch.nn as nn
import sys

class BaselineTextModel(nn.Module):
    def __init__(self, vocab_sz=3000, n_hidden=100):
        super(BaselineTextModel, self).__init__()

        self.embedding = nn.Embedding(vocab_sz, n_hidden)

        self.rnn = nn.LSTM(n_hidden, n_hidden)

        self.dropout = nn.Dropout(0.2)

        self.linear = nn.Linear(n_hidden*2, 3)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)

        x = self.dropout(x)

        # Using the avg and max pool of all RNN outputs
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, 1)

        # We concatenate them (hidden size before the linear layer is multiplied by 2)
        out = torch.cat((avg_pool, max_pool), dim=1)
        out = self.linear(out)

        # We dont apply sigmoid to output since nn.BCEWithLogitsLoss
        # combines a Sigmoid layer and the BCELoss
        return torch.squeeze(out, dim=1)

if __name__=='__main__':
    print('UNIT TEST BaselineTextModel:')
    x=torch.ones(32, 128, dtype=torch.long)
    model = BaselineTextModel()
    out=model(x)
    print('\t out shape: ', out.size())
    print('BaselineTextModel test PASSED')