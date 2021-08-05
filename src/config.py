import os

# CONFIG
RUN_NAME='test2'
BATCH_SIZE=64
WORKER_COUNT=12
EPOCHS=1
DATASET='audio'
MODEL='baseline_audio_model_1'
LOSS='CrossEntropyLoss'
LR=1e-3
OPTIMIZER='Adam'

# CONSTANTS
DATA_FOLDER_PTH=os.path.join(os.path.dirname(__file__), os.pardir, 'data')
RUNS_FOLDER_PTH=os.path.join(os.path.dirname(__file__), os.pardir, 'runs')

TRAIN_AUDIO_FOLDER_PTH=os.path.join(DATA_FOLDER_PTH, 'processed','MELD','train_wavs')
DEV_AUDIO_FOLDER_PTH=os.path.join(DATA_FOLDER_PTH, 'processed','MELD','dev_wavs')
TEST_AUDIO_FOLDER_PTH=os.path.join(DATA_FOLDER_PTH, 'processed','MELD','test_wavs')

TRAIN_TEXT_FILE_PTH=os.path.join(DATA_FOLDER_PTH, 'raw','MELD','train', 'train_sent_emo.csv')
DEV_TEXT_FILE_PTH=os.path.join(DATA_FOLDER_PTH, 'raw','MELD','dev', 'dev_sent_emo.csv')
TEST_TEXT_FILE_PTH=os.path.join(DATA_FOLDER_PTH, 'raw','MELD','test', 'test_sent_emo.csv')
