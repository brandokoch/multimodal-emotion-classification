import pandas as pd
import tensorflow as tf
import numpy as np
import config
import re
import wandb
import string


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

        data = pd.read_csv(wandb.config.TRAIN_TEXT_FILE_PTH)
        text = data['Utterance']
        #print(text)

        # Normalize texts
        def normalize(string_list):
            re_print = re.compile('[^%s]' % re.escape(string.printable))
            normalized_string_list = []
            for string_item in string_list:
                normalized_string = ''.join([re_print.sub('', w) for w in string_item])
                normalized_string_list.append(normalized_string)
            return normalized_string_list

        text = normalize(text)

        # Preprocessing
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=3000, filters='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(text)

        text = tokenizer.texts_to_sequences(text)
        text = tf.keras.preprocessing.sequence.pad_sequences(text, maxlen=wandb.config.TEXT_MAX_LENGTH)
        #print(f'Sample sentence tokenized: {text[0]}, shape: {text[0].shape}')

        embedding_matrix = self.create_embedding_matrix(tokenizer.word_index, embedding_dict=glove_embedding, dimension=100)
        #print(embedding_matrix.shape)

        return embedding_matrix

    def __call__(self):
        return self.embedding_matrix


import torch
import torch.nn as nn

if __name__=='__main__':
    print('UNIT TEST GLOVE_LOADING:')
    gwl = GloveWeightsLoading()
    embedding_matrix = gwl()

    vocab_size = embedding_matrix.shape[0]
    vector_size = embedding_matrix.shape[1]

    embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vector_size)

    embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

    print(embedding(torch.LongTensor([1])))

    print('GLOVE_LOADING test PASSED')