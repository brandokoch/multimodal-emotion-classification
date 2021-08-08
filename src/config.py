import os
import torch
import wandb

# RUN CONFIG
RUN_NAME='text7_transformer_1head_2layer'
BATCH_SIZE=32
WORKER_COUNT=0
EPOCHS=60
DATASET='text'
MODEL='transformer_text_model'
LOSS='CrossEntropyLoss'
LR=1e-3
OPTIMIZER='Adagrad'

# CLASS_WEIGHTS=torch.tensor([1,2,2,2,2,2,2], dtype=torch.float32)
# CLASS_WEIGHTS=torch.tensor([2,8,37,14,5,36,9], dtype=torch.float32)

# AUDIO CONFIG
STD=torch.tensor(21.01, dtype=torch.float32)
MEAN=torch.tensor(-64.38, dtype=torch.float32)

# TEXT CONFIG
TEXT_MAX_LENGTH=128

# CONSTANTS
DATA_FOLDER_PTH=os.path.join(os.path.dirname(__file__), os.pardir, 'data')
RUNS_FOLDER_PTH=os.path.join(os.path.dirname(__file__), os.pardir, 'runs')
CACHE_FOLDER_PTH=os.path.join(os.path.dirname(__file__), os.pardir, 'data','cache')
MODELS_FOLDER_PTH = os.path.join(os.path.dirname(__file__), os.pardir, 'models')

TRAIN_AUDIO_FOLDER_PTH=os.path.join(DATA_FOLDER_PTH, 'processed','MELD','train_wavs')
DEV_AUDIO_FOLDER_PTH=os.path.join(DATA_FOLDER_PTH, 'processed','MELD','dev_wavs')
TEST_AUDIO_FOLDER_PTH=os.path.join(DATA_FOLDER_PTH, 'processed','MELD','test_wavs')

TRAIN_TEXT_FILE_PTH=os.path.join(DATA_FOLDER_PTH, 'raw','MELD','train', 'train_sent_emo.csv')
DEV_TEXT_FILE_PTH=os.path.join(DATA_FOLDER_PTH, 'raw','MELD','dev', 'dev_sent_emo.csv')
TEST_TEXT_FILE_PTH=os.path.join(DATA_FOLDER_PTH, 'raw','MELD','test', 'test_sent_emo.csv')

GLOVE_MODEL_PATH = os.path.join(MODELS_FOLDER_PTH, 'glove', 'glove.6B.100d.txt')
