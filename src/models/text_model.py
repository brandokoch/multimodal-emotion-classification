import torch
import torch.nn as nn
import sys
import math
from src.gloveWeightsLoading import GloveWeightsLoading
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerTextModel(nn.Module):
    def __init__(self, vocab_sz=3000, n_hidden=100):
        super(TransformerTextModel, self).__init__()

        self.ninp = n_hidden
        gwl = GloveWeightsLoading()
        embedding_matrix = gwl()

        vocab_size = embedding_matrix.shape[0]
        embeddin_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddin_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.pos_encoder = PositionalEncoding(n_hidden, 0.25)

        self.dropout = nn.Dropout(0.25)

        encoder_layers = TransformerEncoderLayer(self.ninp, 1, 100, dropout=0.25)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)

        self.linear = nn.Linear(n_hidden * 2, 7)

        self.src_mask = None

    def forward(self, x):

        out = x.permute(1, 0)

        if self.src_mask is None:
            device = out.device
            mask = self.generate_square_subsequent_mask(len(out)).to(device)
            self.src_mask = mask

        out = self.embedding(out) * math.sqrt(self.ninp)
        out = self.pos_encoder(out)

        out = self.transformer_encoder(out, self.src_mask)

        out = out.permute(1, 0, 2)
        # Using the avg and max pool of all RNN outputs
        avg_pool = torch.mean(out, dim=1)
        max_pool, _ = torch.max(out, dim=1)

        # We concatenate them (hidden size before the linear layer is multiplied by 2)
        out = torch.cat((avg_pool, max_pool), dim=1)
        out = self.linear(out)

        return out

    def generate_square_subsequent_mask(self, sz: int):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class BaselineTextModel(nn.Module):
    def __init__(self, vocab_sz=3000, n_hidden=100):
        super(BaselineTextModel, self).__init__()

        gwl = GloveWeightsLoading()
        embedding_matrix = gwl()

        vocab_size = embedding_matrix.shape[0]
        embeddin_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddin_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.dropout = nn.Dropout(0.25)

        self.rnn = nn.LSTM(input_size = embeddin_dim, hidden_size = n_hidden, num_layers=2, bidirectional=True, batch_first=True)

        self.linear = nn.Linear(n_hidden * 4, n_hidden)
        self.relu = nn.ReLU()
        self.classify = nn.Linear(n_hidden, 7)

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

    print('UNIT TEST TransformerTextModel:')
    x = torch.ones(32, 128, dtype=torch.long)
    model = TransformerTextModel()
    out = model(x)
    print('\t out shape: ', out.size())
    print('BaselineTextModel test PASSED')
