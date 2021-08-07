import wandb

RUN_NAME='test'
wandb.init(project='multimodal-emotion-classification', name=RUN_NAME, entity='psiml7-multimodal-emotion-clf')

#------------------------------------------------------------------------------------------------------------#
# GENERAL CONFIG
wandb.config.RUN_NAME = RUN_NAME
wandb.config.BATCH_SIZE=256
wandb.config.WORKER_COUNT=0
wandb.config.EPOCHS=2
wandb.config.DATASET='multimodal'
wandb.config.MODEL='baseline_multimodal_model'
wandb.config.CBS='multimodal'
wandb.config.LOSS='CrossEntropyLoss'
# OPTIMIZER CONFIG
wandb.config.OPTIMIZER='Adam'
wandb.config.WEIGHT_DECAY=0.0001
wandb.config.LR=1e-6
# SCHEDULER CONFIG
wandb.config.SCHEDULER='OneCycleLR'
wandb.config.SCHEDULER_MAX_LR=1e-3
# TEXT CONFIG
wandb.config.VOCAB_SIZE=3000
wandb.config.TEXT_MAX_LENGTH=128
# GENERAL PATH CONFIG
wandb.config.DATA_FOLDER_PTH="../data"
wandb.config.RUNS_FOLDER_PTH="../runs"
wandb.config.CACHE_FOLDER_PTH="../data/cache"
wandb.config.MODELS_FOLDER_PTH = "../models"
wandb.config.GLOVE_MODEL_PATH = "../models/glove.6B.100d.txt"
# AUDIO PATH CONFIG
wandb.config.TRAIN_AUDIO_FOLDER_PTH="../data/processed/MELD/train_wavs"
wandb.config.DEV_AUDIO_FOLDER_PTH="../data/processed/MELD/dev_wavs"
wandb.config.TEST_AUDIO_FOLDER_PTH="../data/processed/MELD/test_wavs"
# TEXT PATH CONFIG
wandb.config.TRAIN_TEXT_FILE_PTH="../data/raw/MELD/train/train_sent_emo.csv"
wandb.config.DEV_TEXT_FILE_PTH="../data/raw/MELD/dev/dev_sent_emo.csv"
wandb.config.TEST_TEXT_FILE_PTH="../data/raw/MELD/test/test_sent_emo.csv"
#------------------------------------------------------------------------------------------------------------#