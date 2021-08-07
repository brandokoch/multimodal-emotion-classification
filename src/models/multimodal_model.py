import torch
from torch import nn
import torch.nn.functional as F
import config
import os
import io
import json
import numpy as np
from glove_loading import GloveWeightsLoading


# def load_vectors(fname):
#     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())
#     data = {}
#     for line in fin:
#         tokens = line.rstrip().split(' ')
#         data[tokens[0]] = map(float, tokens[1:])
#     return data



# def create_embedding_matrix(word_index, embedding_dict):
#     """
#     This function creates the embedding matrix.
#     :param word index: a dictionary with word:index_value
#     :param embedding_dict: a dictionary with word:embedding_vector
#     :return: a numpy array with embedding vectors for all know words
#     """

#     embedding_matrix=np.zeros((len(word_index)+1, 300))
#     for word, i in word_index.items():
#         if word in embedding_dict:
#             embedding_matrix[i] = embedding_dict[word]
#     return embedding_matrix


class BaselineMultimodalModel(nn.Module):
    def __init__(self, n_hidden=100, n_input=1, n_output=3, stride=16, n_channel=32):
        super(BaselineMultimodalModel, self).__init__()

        # Pretrained embedding loading
        # tokenizer_word_index=json.load(open(os.path.join(config.RUNS_FOLDER_PTH, config.RUN_NAME, 'word_index.json'),'r'))
        # embedding_dict = load_vectors(os.path.join(config.MODELS_FOLDER_PTH, 'crawl-300d-2M-subword.vec'))
        # embedding_matrix=create_embedding_matrix(tokenizer_word_index, embedding_dict)
        # num_words=embedding_matrix.shape[0]
        # embed_dim=embedding_matrix.shape[1]


        # Text Encoder parts:
        gwl = GloveWeightsLoading()
        embedding_matrix = gwl()
        vocab_size = embedding_matrix.shape[0]
        embeddin_dim = embedding_matrix.shape[1]
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddin_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        # self.embedding=nn.Embedding(
        #     num_embeddings=num_words,
        #     embeddding_dim=embed_dim
        # )

        # self.embedding.weight=nn.Parameter(
        #     torch.tensor(
        #         embedding_matrix,
        #         dtype=torch.float32
        #     )
        # )

        self.embedding.weight.requires_grad=False

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


if __name__=='__main__':
    print('UNIT TEST MultimodalBaseline:')
    x_1=torch.zeros(32, 128, dtype=torch.long)
    x_2=torch.zeros(32, 1, 114518)
    x=(x_1,x_2)
    model = BaselineMultimodalModel()
    out=model(x)
    print('\t out shape: ', out.size())
    print('AudioBaselineModel_3 test PASSED')