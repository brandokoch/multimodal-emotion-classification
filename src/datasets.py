import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob
import librosa
import os
import numpy as np

import config

def get_dataloaders(name):
    if name=='audio':
        return get_audio_dataloaders()
    # if name=='text':
    #     return get_text_dataloaders()
    # if name=='multimodal':
    #     return get_multimodal_datalodaers()


class AudioDataset(Dataset):
    def __init__(self, pths, wav_to_label, max_length=10, sr=8000):
        self.pths=pths
        self.wav_to_label=wav_to_label
        self.sr=sr
        self.max_length=max_length
        self.input_size=157 #fix this
        

    def __len__(self):
        return len(self.pths)

    def __getitem__(self, idx):
        sample, sample_rate=librosa.load(self.pths[idx], sr=self.sr)
        short_samples=librosa.util.fix_length(sample, self.sr * self.max_length)
        melSpectrum=librosa.feature.melspectrogram(short_samples.astype(np.float16), sr=self.sr, n_mels=40)
        logMelSpectrum=librosa.power_to_db(melSpectrum, ref=np.max)

        logMelSpectrum=torch.flatten(torch.tensor(logMelSpectrum))
        label=torch.tensor(self.wav_to_label[self.pths[idx].split('\\')[-1]], dtype=torch.long)
        return logMelSpectrum, label

def get_audio_dataloaders():
    # pths
    org_train_audio_pths=glob.glob(os.path.join(config.TRAIN_AUDIO_FOLDER_PTH, '*.wav'))

    # making train and dev out of org_train
    split_idx=int(len(org_train_audio_pths)*0.8)
    train_audio_pths=org_train_audio_pths[:split_idx]
    val_audio_pths=org_train_audio_pths[split_idx:]

    train_text=pd.read_csv(config.TRAIN_TEXT_FILE_PTH)

    def info_to_wav_name(dialogue_id, utterance_id):
        return 'dia{}_utt{}.wav'.format(dialogue_id, utterance_id)

    def emotion_to_label(emotion):
        if emotion=='neutral':
            return 0
        elif emotion=='surprise':
            return 1
        elif emotion=='fear':
            return 2
        elif emotion=='sadness':
            return 3
        elif emotion=='joy':
            return 4
        elif emotion=='disgust':
            return 5
        elif emotion=='anger':
            return 6

    train_text['wav_name']=train_text.apply(lambda x: info_to_wav_name(x['Dialogue_ID'], x['Utterance_ID']), axis=1)
    train_text['label']=train_text.apply(lambda x: emotion_to_label(x['Emotion']), axis=1)

    wav_to_label=dict(zip(train_text['wav_name'], train_text['label']))

    train_ds=AudioDataset(train_audio_pths, wav_to_label)
    val_ds=AudioDataset(val_audio_pths, wav_to_label)

    train_loader=DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.WORKER_COUNT)
    val_loader=DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.WORKER_COUNT)

    return train_loader, val_loader

if __name__=='__main__':
    print('UNIT TEST:')
    train_loader, val_loader=get_audio_dataloaders()
    print('\t Audio dataset X shape ',next(iter(train_loader))[0].size())
    print('\t Audio dataset y shape', next(iter(train_loader))[1].size())
    print('Audio Unit test PASSED')