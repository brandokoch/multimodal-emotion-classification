import pandas as pd
import tensorflow as tf
import numpy as np
import re
import string
import wandb
import json
import os
from keras_preprocessing.text import tokenizer_from_json


class GloveWeightsLoading:
    def __init__(self):
        self.embedding_matrix = self.getWeights()

    def create_embedding_matrix(self, word_index, embedding_dict, dimension):
        embedding_matrix = np.zeros((len(word_index) + 1, dimension))

        for word, index in word_index.items():
            if word in embedding_dict:
                embedding_matrix[index] = embedding_dict[word]
        return embedding_matrix

    def getWeights(self):
        glove = pd.read_csv(wandb.config.GLOVE_MODEL_PATH, sep=" ", quoting=3, header=None, index_col=0)
        glove_embedding = {key: val.values for key, val in glove.T.items()}
        #print(glove_embedding['cat'])


        tok_pth=os.path.join(wandb.config.RUNS_FOLDER_PTH,wandb.config.RUN_NAME, wandb.config.MODEL+'_tok.json')
        with open(tok_pth) as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)

        embedding_matrix = self.create_embedding_matrix(tokenizer.word_index, embedding_dict=glove_embedding, dimension=100)
        #print(embedding_matrix.shape)

        return embedding_matrix

    def __call__(self):
        return self.embedding_matrix


import torch
import torch.nn as nn

if __name__=='__main__':
    import config

    print('UNIT TEST GLOVE_LOADING:')
    gwl = GloveWeightsLoading()
    embedding_matrix = gwl()

    vocab_size = embedding_matrix.shape[0]
    vector_size = embedding_matrix.shape[1]

    embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vector_size)

    embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

    print(embedding(torch.LongTensor([1])))

    print('GLOVE_LOADING test PASSED')