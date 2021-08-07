import torch
import torch.nn as nn
import sys
from src.gloveWeightsLoading import GloveWeightsLoading

class BaselineTextModel(nn.Module):
    def __init__(self, vocab_sz=3000, n_hidden=100):
        super(BaselineTextModel, self).__init__()

        gwl = GloveWeightsLoading()
        embedding_matrix = gwl()

        vocab_size = embedding_matrix.shape[0]
        embeddin_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddin_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        #self.embedding.weight.requires_grad = False

        self.dropout = nn.Dropout(0.25)

        self.rnn = nn.LSTM(input_size = embeddin_dim, hidden_size = n_hidden, num_layers=2, bidirectional=True)

        self.linear = nn.Linear(n_hidden * 4, n_hidden)
        self.relu = nn.ReLU()
        self.classify = nn.Linear(n_hidden, 3)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)

        x, _ = self.rnn(x)
        x = self.dropout(x)

        # Using the avg and max pool of all RNN outputs
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, 1)

        # We concatenate them (hidden size before the linear layer is multiplied by 2)
        out = torch.cat((avg_pool, max_pool), dim=1)

        out = self.linear(out)

        #out = self.linear(x[:,-1,:])


        out = self.relu(out)
        out = self.dropout(out)


        out = self.classify(out)

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
