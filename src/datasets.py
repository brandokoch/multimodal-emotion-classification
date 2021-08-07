from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob
import librosa
import os
import json
import numpy as np
import config
import re
import torchaudio
import string
import concurrent
import io
from torchaudio.functional import resample
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def get_dataloaders(name):
    if name=='audio':
        return get_audio_dataloaders()
    if name=='text':
        return get_text_dataloaders()
    if name=='multimodal':
        return get_multimodal_dataloaders()


# class AudioDataset(Dataset):
#     def __init__(self, pths, wav_to_label, max_length=15, sr=8000, recalculate=False):
#         self.pths=pths
#         self.wav_to_label=wav_to_label
#         self.sr=sr
#         self.max_length=max_length
#         self.input_size=157 #fix this

#         # cache
#         if recalculate:
#             executor=concurrent.futures.ProcessPoolExecutor(max_workers=10)
#             futures=[executor.submit(self.preprocess_and_cache_file, pth) for pth in self.pths]
#             concurrent.futures.wait(futures)


#     def preprocess_and_cache_file(self,pth):
#         sample, sample_rate=librosa.load(pth, sr=self.sr)
#         short_samples=librosa.util.fix_length(sample, self.sr * self.max_length)
#         melSpectrum=librosa.feature.melspectrogram(short_samples.astype(np.float16), sr=self.sr, n_mels=128)
#         logMelSpectrum=librosa.power_to_db(melSpectrum, ref=np.max)
#         label=self.wav_to_label[pth.split('\\')[-1]]

#         cache_pth=os.path.join(config.CACHE_FOLDER_PTH, pth.split('\\')[-1].split('.')[0]+'.npy')
#         np.save(cache_pth, (logMelSpectrum,label))


#     def __len__(self):
#         return len(self.pths)

#     def __getitem__(self, idx):
        
#         cache_pth=os.path.join(config.CACHE_FOLDER_PTH, self.pths[idx].split('\\')[-1].split('.')[0]+'.npy')
#         logMelSpectrum, label=np.load(cache_pth, allow_pickle=True)

#         logMelSpectrum=torch.unsqueeze(torch.tensor(logMelSpectrum),0)
#         label=torch.tensor(label,dtype=torch.long)

#         # normalize 
#         logMelSpectrum=(logMelSpectrum-config.MEAN)/config.STD

#         return logMelSpectrum, label


# def get_audio_dataloaders():
#     # pths
#     org_train_audio_pths=glob.glob(os.path.join(config.TRAIN_AUDIO_FOLDER_PTH, '*.wav'))

#     # making train and dev out of org_train
#     split_idx=int(len(org_train_audio_pths)*0.8)
#     train_audio_pths=org_train_audio_pths[:split_idx]
#     val_audio_pths=org_train_audio_pths[split_idx:]

#     train_text=pd.read_csv(config.TRAIN_TEXT_FILE_PTH)

#     def info_to_wav_name(dialogue_id, utterance_id):
#         return 'dia{}_utt{}.wav'.format(dialogue_id, utterance_id)

#     # def emotion_to_label(emotion):
#     #     if emotion=='neutral':
#     #         return 0
#     #     elif emotion=='surprise':
#     #         return 1
#     #     elif emotion=='fear':
#     #         return 2
#     #     elif emotion=='sadness':
#     #         return 3
#     #     elif emotion=='joy':
#     #         return 4
#     #     elif emotion=='disgust':
#     #         return 5
#     #     elif emotion=='anger':
#     #         return 6

#     def emotion_to_label(emotion):
#         if emotion=='neutral':
#             return 0
#         elif emotion=='positive':
#             return 1
#         elif emotion=='negative':
#             return 2

#     # train_text['wav_name']=train_text.apply(lambda x: info_to_wav_name(x['Dialogue_ID'], x['Utterance_ID']), axis=1)
#     # train_text['label']=train_text.apply(lambda x: emotion_to_label(x['Emotion']), axis=1)

#     train_text['wav_name']=train_text.apply(lambda x: info_to_wav_name(x['Dialogue_ID'], x['Utterance_ID']), axis=1)
#     train_text['label']=train_text.apply(lambda x: emotion_to_label(x['Sentiment']), axis=1)

#     wav_to_label=dict(zip(train_text['wav_name'], train_text['label']))

#     train_ds=AudioDataset(train_audio_pths, wav_to_label)
#     val_ds=AudioDataset(val_audio_pths, wav_to_label)

#     train_loader=DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.WORKER_COUNT)
#     val_loader=DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.WORKER_COUNT)

#     return train_loader, val_loader

class AudioDataset(Dataset):
    def __init__(self, df, folder_pth):
        self.df=df
        self.pths=df.wav_name.values
        self.folder_pth=folder_pth
        self.labels=df.label.values

    def __len__(self):
        return len(self.pths)

    def __getitem__(self, idx):
        waveform, sr=torchaudio.load(os.path.join(self.folder_pth,self.pths[idx]))
        label=self.labels[idx]

        # To mono
        waveform=torch.mean(waveform, dim=0).unsqueeze(0)

        # Resample        
        waveform=resample(waveform,orig_freq=sr, new_freq=8000)

        # To tensors
        label=torch.tensor(label, dtype= torch.long)

        return (waveform, label)

def get_audio_dataloaders():
    train_df=pd.read_csv(config.TRAIN_TEXT_FILE_PTH)[['Dialogue_ID','Utterance_ID','Sentiment']]
    val_df=pd.read_csv(config.DEV_TEXT_FILE_PTH)[['Dialogue_ID','Utterance_ID','Sentiment']]

    def info_to_wav_name(dialogue_id, utterance_id):
        return 'dia{}_utt{}.wav'.format(dialogue_id, utterance_id)

    def emotion_to_label(emotion):
        if emotion=='neutral':
            return 0
        elif emotion=='positive':
            return 1
        elif emotion=='negative':
            return 2

    train_df['wav_name']=train_df.apply(lambda x: info_to_wav_name(x['Dialogue_ID'], x['Utterance_ID']), axis=1)
    train_df['label']=train_df.apply(lambda x: emotion_to_label(x['Sentiment']), axis=1)
    train_df=train_df[['wav_name','label']]

    val_df['wav_name']=val_df.apply(lambda x: info_to_wav_name(x['Dialogue_ID'], x['Utterance_ID']), axis=1)
    val_df['label']=val_df.apply(lambda x: emotion_to_label(x['Sentiment']), axis=1)
    val_df=val_df[['wav_name','label']]

    # drop row where wav_name is dia125_utt3.wav (WARNING) 
    train_df=train_df[train_df['wav_name']!='dia125_utt3.wav']
    val_df=val_df[val_df['wav_name']!='dia110_utt7.wav']
    # val_df=val_df[val_df['wav_name']!='dia23_utt13.wav']


    train_ds=AudioDataset(train_df, config.TRAIN_AUDIO_FOLDER_PTH)
    val_ds=AudioDataset(val_df, config.DEV_AUDIO_FOLDER_PTH)

    def pad_sequences(batch):
        batch=[item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0,2,1)

    def collate_fn(batch):
        
        tensors, targets=[], []
        for waveform, label in batch:
            tensors+=[waveform]
            targets+=[label]
        
        tensors=pad_sequences(tensors)
        targets=torch.stack(targets)

        return tensors, targets


    train_loader=DataLoader(train_ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, pin_memory=True, shuffle=True, num_workers=config.WORKER_COUNT)
    val_loader=DataLoader(val_ds, batch_size=config.BATCH_SIZE,collate_fn=collate_fn, pin_memory=True, shuffle=True, num_workers=config.WORKER_COUNT, drop_last=False)

    return train_loader, val_loader


class TextDataset(Dataset):
    def __init__(self,texts,targets):
        self.texts=texts
        self.targets=targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self,idx):

        text=self.texts[idx]
        target=self.targets[idx]

        text=torch.tensor(text,dtype=torch.long)
        target=torch.tensor(target,dtype=torch.long)

        item=(text,
              target)

        return item

def get_text_dataloaders():
        data=pd.read_csv(config.TRAIN_TEXT_FILE_PTH) # using only train set
        data=data.sample(frac=1, random_state=42)

        # Loading data
        train=data[:7000]
        xtrain=train['Utterance'].values.tolist()
        ytrain=train['Sentiment'].values

        val=data[7000:]
        xval=val['Utterance'].values.tolist()
        yval=val['Sentiment'].values

        # Normalize texts
        def normalize(string_list):
            re_print=re.compile('[^%s]' % re.escape(string.printable))
            normalized_string_list=[]
            for string_item in string_list:
                normalized_string=''.join([re_print.sub('',w) for w in string_item])
                normalized_string_list.append(normalized_string)
            return normalized_string_list

        xtrain=normalize(xtrain)
        xval=normalize(xval)

        # Preprocessing
        tokenizer=tf.keras.preprocessing.text.Tokenizer(num_words=config.VOCAB_SIZE, filters='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(xtrain)

        # tokenizer_json = tokenizer.to_json()
        # pth=os.path.join(config.RUNS_FOLDER_PTH,config.RUN_NAME, config.MODEL+'_tok.json')
        # with io.open(pth, 'w', encoding='utf-8') as f:
        #     f.write(json.dumps(tokenizer_json, ensure_ascii=False))

        xtrain_pro=tokenizer.texts_to_sequences(xtrain)
        xtrain_pro=tf.keras.preprocessing.sequence.pad_sequences(xtrain_pro, maxlen=config.TEXT_MAX_LENGTH)

        xval_pro=tokenizer.texts_to_sequences(xval)
        xval_pro=tf.keras.preprocessing.sequence.pad_sequences(xval_pro, maxlen=config.TEXT_MAX_LENGTH)

        def emotion_to_label(emotion):
            if emotion=='neutral':
                return 0
            elif emotion=='positive':
                return 1
            elif emotion=='negative':
                return 2

        ytrain=[emotion_to_label(y) for y in ytrain]
        yval=[emotion_to_label(y) for y in yval]

        # Creating Datasets
        train_ds=TextDataset(xtrain_pro, ytrain)
        val_ds=TextDataset(xval_pro, yval)

        # Creating DataLoaders
        train_dl=torch.utils.data.DataLoader(
                train_ds,
                batch_size=config.BATCH_SIZE,
                num_workers=config.WORKER_COUNT,
                )

        val_dl=torch.utils.data.DataLoader(
                val_ds,
                batch_size=config.BATCH_SIZE,
                num_workers=config.WORKER_COUNT,
                )

        return train_dl, val_dl


class MultimodalDataset(Dataset):
    def __init__(self,texts,targets, pths, folder_pth):
        self.texts=texts
        self.targets=targets
        self.pths=pths
        self.folder_pth=folder_pth

    def __len__(self):
        return len(self.texts)

    def __getitem__(self,idx):

        text=self.texts[idx]
        target=self.targets[idx]
        waveform, sr=torchaudio.load(os.path.join(self.folder_pth,self.pths[idx]))

        # Preprocess audio
        waveform=torch.mean(waveform, dim=0).unsqueeze(0)
        waveform=resample(waveform,orig_freq=sr, new_freq=8000)

        # To tensors
        text=torch.tensor(text,dtype=torch.long)
        target=torch.tensor(target,dtype=torch.long)

        item=((text, waveform), target)
        return item

def get_multimodal_dataloaders():

        def info_to_wav_name(dialogue_id, utterance_id):
            return 'dia{}_utt{}.wav'.format(dialogue_id, utterance_id)

        train_df=pd.read_csv(config.TRAIN_TEXT_FILE_PTH) 
        train_df['wav_name']=train_df.apply(lambda x: info_to_wav_name(x['Dialogue_ID'], x['Utterance_ID']), axis=1)

        val_df=pd.read_csv(config.DEV_TEXT_FILE_PTH) 
        val_df['wav_name']=val_df.apply(lambda x: info_to_wav_name(x['Dialogue_ID'], x['Utterance_ID']), axis=1)

        # drop row where wav_name is dia125_utt3.wav (WARNING) 
        train_df=train_df[train_df['wav_name']!='dia125_utt3.wav']
        val_df=val_df[val_df['wav_name']!='dia110_utt7.wav']

        # Loading data
        xtrain=train_df['Utterance'].values.tolist()
        ytrain=train_df['Sentiment'].values
        pths_train=train_df['wav_name'].values

        xval=val_df['Utterance'].values.tolist()
        yval=val_df['Sentiment'].values
        pths_val=val_df['wav_name'].values

        # Normalize texts
        def normalize(string_list):
            re_print=re.compile('[^%s]' % re.escape(string.printable))
            normalized_string_list=[]
            for string_item in string_list:
                normalized_string=''.join([re_print.sub('',w) for w in string_item.lower()])
                normalized_string_list.append(normalized_string)
            return normalized_string_list

        xtrain=normalize(xtrain)
        xval=normalize(xval)

        # Preprocessing
        tokenizer=tf.keras.preprocessing.text.Tokenizer(num_words=config.VOCAB_SIZE, filters='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(xtrain)

        # Save word index
        with open(os.path.join(config.RUNS_FOLDER_PTH, config.RUN_NAME, 'word_index.json'), 'w') as f:
            json.dump(tokenizer.word_index, f)

        # tokenizer_json = tokenizer.to_json()
        # pth=os.path.join(config.RUNS_FOLDER_PTH,config.RUN_NAME, config.MODEL+'_tok.json')
        # with io.open(pth, 'w', encoding='utf-8') as f:
        #     f.write(json.dumps(tokenizer_json, ensure_ascii=False))

        xtrain_pro=tokenizer.texts_to_sequences(xtrain)
        xtrain_pro=tf.keras.preprocessing.sequence.pad_sequences(xtrain_pro, maxlen=config.TEXT_MAX_LENGTH)

        xval_pro=tokenizer.texts_to_sequences(xval)
        xval_pro=tf.keras.preprocessing.sequence.pad_sequences(xval_pro, maxlen=config.TEXT_MAX_LENGTH)

        def emotion_to_label(emotion):
            if emotion=='neutral':
                return 0
            elif emotion=='positive':
                return 1
            elif emotion=='negative':
                return 2

        ytrain=[emotion_to_label(y) for y in ytrain]
        yval=[emotion_to_label(y) for y in yval]

        # Creating Datasets
        train_ds=MultimodalDataset(xtrain_pro, ytrain, pths_train, config.TRAIN_AUDIO_FOLDER_PTH)
        val_ds=MultimodalDataset(xval_pro, yval, pths_val, config.DEV_AUDIO_FOLDER_PTH)

        # Creating DataLoaders
        def pad_sequences(batch):
            batch=[item.t() for item in batch]
            batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
            return batch.permute(0,2,1)

        def collate_fn(batch):
            
            texts, waveforms, targets=[], [], []
            for (text, waveform), label in batch:
                texts+=[text]
                waveforms+=[waveform]
                targets+=[label]
            
            waveforms=pad_sequences(waveforms)
            texts=torch.stack(texts)
            targets=torch.stack(targets)

            return (texts, waveforms), targets


        train_loader=DataLoader(train_ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, pin_memory=True, shuffle=True, num_workers=config.WORKER_COUNT)
        val_loader=DataLoader(val_ds, batch_size=config.BATCH_SIZE,collate_fn=collate_fn, pin_memory=True, shuffle=True, num_workers=config.WORKER_COUNT, drop_last=False)

        return train_loader, val_loader

if __name__=='__main__':
    # print('AUDIO UNIT TEST:')
    # train_loader, val_loader=get_audio_dataloaders()
    # print('\t AUDIO dataset X shape ',next(iter(train_loader))[0])
    # print('\t AUDIO dataset X shape ',next(iter(train_loader))[0].size())
    # print('\t AUDIO dataset y shape', next(iter(train_loader))[1].size())
    # print('AUDIO Unit test PASSED')

    # print('TEXT UNIT TEST:')
    # train_loader, val_loader=get_text_dataloaders()
    # # print('\t TEXT dataset X shape ',next(iter(train_loader))[0])
    # print('\t TEXT dataset X shape ',next(iter(train_loader))[0].size())
    # print('\t TEXT dataset y shape', next(iter(train_loader))[1].size())
    # print('Text Unit test PASSED')

    print('MULTIMODAL UNIT TEST:')
    train_loader, val_loader=get_multimodal_dataloaders()
    print('\t MULTIMODAL dataset sample',next(iter(train_loader))[0][0])
    print('\t MULTIMODAL dataset Text shape ',next(iter(train_loader))[0][0].size())
    print('\t MULTIMODAL dataset Waveform shape ',next(iter(train_loader))[0][1].size())
    print('\t MULTIMODAL dataset y shape', next(iter(train_loader))[2].size())
    print('MULTIMODAL Unit test PASSED')