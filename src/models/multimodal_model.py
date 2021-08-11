import torch
from torch import nn
import torch.nn.functional as F
import os
import io
import json
import numpy as np
import wandb
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from gloveWeightsLoading import GloveWeightsLoading


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

    def __init__(self, d_model, dropout=0.1, max_len=10000):
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


class MultimodalModel(nn.Module):
    def __init__(self, n_hidden=100, n_input=1, n_output=7, stride=16, n_channel=32):
        super(MultimodalModel, self).__init__()

        # Text Encoder parts
        self.embedding = nn.Embedding(num_embeddings=wandb.config.VOCAB_SIZE, embedding_dim=n_hidden)
        self.rnn = nn.LSTM(n_hidden, n_hidden)
        self.dropout = nn.Dropout(0.2)

        # Audio Encoder parts
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)
        self.flatten=nn.Flatten()

        # Multimodal Decoder parts:
        self.lin1 = nn.Linear(264, 100)
        self.lin2 = nn.Linear(100, n_output)


    def forward(self,x):
        #
        # Input seperation
        # (x is[text_batch, audio_batch] )
        # 
        text_h=x[0]
        audio_h=x[1]
        
        #
        # Text Encoder:
        #

        text_h = self.embedding(text_h)
        text_h, _ = self.rnn(text_h)
        text_h = self.dropout(text_h)

        text_avg_pool = torch.mean(text_h, dim=1)
        text_max_pool, _ = torch.max(text_h, 1)

        text_h = torch.cat((text_avg_pool, text_max_pool), dim=1) #32x200

        #
        # Audio Encoder:
        #
        audio_h = self.conv1(audio_h)
        audio_h = F.relu(self.bn1(audio_h))
        audio_h = self.pool1(audio_h)
        audio_h = self.conv2(audio_h)
        audio_h = F.relu(self.bn2(audio_h))
        audio_h = self.pool2(audio_h)
        audio_h = self.conv3(audio_h)
        audio_h = F.relu(self.bn3(audio_h))
        audio_h = self.pool3(audio_h)
        audio_h = self.conv4(audio_h)
        audio_h = F.relu(self.bn4(audio_h))
        audio_h = self.pool4(audio_h)
        audio_h = F.avg_pool1d(audio_h, audio_h.shape[-1])
        audio_h = audio_h.permute(0, 2, 1)
        audio_h= self.flatten(audio_h) #32x64

        #
        # Multimodal connection:
        #
        multimodal_h=torch.cat((text_h, audio_h), dim=1) #32x200+32x64=32x264
        multimodal_h=self.lin1(multimodal_h)
        multimodal_h=F.relu(multimodal_h)
        multimodal_h=self.lin2(multimodal_h)

        return multimodal_h


class MultimodalModel2(nn.Module):
    def __init__(self, n_hidden=100, n_input=1, n_output=7, stride=16, n_channel=32):
        super(MultimodalModel2, self).__init__()

        # Text encoder parts
        self.ninp = n_hidden
        gwl = GloveWeightsLoading()
        embedding_matrix = gwl()

        vocab_size = embedding_matrix.shape[0]
        embeddin_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddin_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding = nn.Embedding(num_embeddings=10000, embedding_dim=n_hidden)

        self.pos_encoder = PositionalEncoding(n_hidden, 0.25)
        self.dropout = nn.Dropout(0.25)

        encoder_layers = TransformerEncoderLayer(self.ninp, 1, 100, dropout=0.25)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)

        self.linear = nn.Linear(n_hidden * 2, n_output )
        self.src_mask = None

        # Audio Encoder parts
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)
        self.flatten=nn.Flatten()

        # Multimodal Decoder parts:
        self.lin1 = nn.Linear(264, 100)
        self.lin2 = nn.Linear(100, n_output)


    def forward(self,x):
        #
        # Input seperation
        # (x is[text_batch, audio_batch] )
        # 
        text_h=x[0]
        audio_h=x[1]
        
        #
        # Text Encoder:
        #

        text_h = text_h.permute(1, 0)

        if self.src_mask is None:
            device = text_h.device
            mask = self.generate_square_subsequent_mask(len(text_h)).to(device)
            self.src_mask = mask

        text_h = self.embedding(text_h) * math.sqrt(self.ninp)
        text_h = self.pos_encoder(text_h)
        text_h = self.transformer_encoder(text_h, self.src_mask)

        text_h = text_h.permute(1, 0, 2)
        # Using the avg and max pool of all outputs
        avg_pool = torch.mean(text_h, dim=1)
        max_pool, _ = torch.max(text_h, dim=1)

        # We concatenate them (hidden size before the linear layer is multiplied by 2)
        text_h = torch.cat((avg_pool, max_pool), dim=1)


        #
        # Audio Encoder:
        #
        audio_h = self.conv1(audio_h)
        audio_h = F.relu(self.bn1(audio_h))
        audio_h = self.pool1(audio_h)
        audio_h = self.conv2(audio_h)
        audio_h = F.relu(self.bn2(audio_h))
        audio_h = self.pool2(audio_h)
        audio_h = self.conv3(audio_h)
        audio_h = F.relu(self.bn3(audio_h))
        audio_h = self.pool3(audio_h)
        audio_h = self.conv4(audio_h)
        audio_h = F.relu(self.bn4(audio_h))
        audio_h = self.pool4(audio_h)
        audio_h = F.avg_pool1d(audio_h, audio_h.shape[-1])
        audio_h = audio_h.permute(0, 2, 1)
        audio_h= self.flatten(audio_h) #32x64

        #
        # Multimodal connection:
        #
        multimodal_h=torch.cat((text_h, audio_h), dim=1) #32x200+32x64=32x264
        multimodal_h=self.lin1(multimodal_h)
        multimodal_h=F.relu(multimodal_h)
        multimodal_h=self.lin2(multimodal_h)

        return multimodal_h

    def generate_square_subsequent_mask(self, sz: int):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)



if __name__=='__main__':

    # --- Fix ---
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 
    import config
    from gloveWeightsLoading import GloveWeightsLoading
    # -----------

    print('UNIT TEST Multimodal model 2:')
    x_1=torch.zeros(32, 128, dtype=torch.long)
    x_2=torch.zeros(32, 1, 114518)
    x=(x_1,x_2)
    model = MultimodalModel2()
    out=model(x)
    print('\t out shape: ', out.size())
    print('Multimodal model 2 test PASSED')