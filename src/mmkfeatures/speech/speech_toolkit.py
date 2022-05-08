import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import librosa
import os

class SpeechEmotionToolkit:
    def __init__(self,model_path,model_para_path,VOICE_LEN=32000):
        self.VOICE_LEN = VOICE_LEN
        self.model_path=model_path
        self.model_para_path=model_para_path

    def normalizeVoiceLen(self,y, normalizedLen):
        nframes = len(y)
        y = np.reshape(y, [nframes, 1]).T
        # 归一化音频长度为2s,32000数据点
        if (nframes < normalizedLen):
            res = normalizedLen - nframes
            res_data = np.zeros([1, res], dtype=np.float32)
            y = np.reshape(y, [nframes, 1]).T
            y = np.c_[y, res_data]
        else:
            y = y[:, 0:normalizedLen]
        return y[0]

    def getNearestLen(self,framelength, sr):
        framesize = framelength * sr
        # 找到与当前framesize最接近的2的正整数次方
        nfftdict = {}
        lists = [32, 64, 128, 256, 512, 1024]
        for i in lists:
            nfftdict[i] = abs(framesize - i)
        sortlist = sorted(nfftdict.items(), key=lambda x: x[1])  # 按与当前framesize差值升序排列
        framesize = int(sortlist[0][0])  # 取最接近当前framesize的那个2的正整数次方值为新的framesize
        return framesize

    def load(self):
        # Load Model
        self.model = load_model(self.model_path)
        self.paradict = {}
        with open(self.model_para_path, 'rb') as f:
            paradict = pickle.load(f)
        self.DATA_MEAN = paradict['mean']
        self.DATA_STD = paradict['std']
        self.emotionDict = paradict['emotion']
        self.edr = dict([(i, t) for t, i in self.emotionDict.items()])

    def analyze_block(self,y,sr):

        N_FFT = self.getNearestLen(0.25, sr)

        y = self.normalizeVoiceLen(y, self.VOICE_LEN)  # 归一化长度

        mfcc_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=int(N_FFT / 4))
        feature = np.mean(mfcc_data, axis=0)
        feature = feature.reshape((126, 1))
        feature -= self.DATA_MEAN
        feature /= self.DATA_STD
        feature = feature.reshape((1, 126, 1))
        result = self.model.predict(feature)
        index = np.argmax(result, axis=1)[0]
        # print(self.edr[index])
        return self.edr[index]

    def analyze(self,wav_file):

        # Test

        # test_path = r'RawData/CASIA database/liuchanhg/angry/203.wav'

        y, sr = librosa.load(wav_file, sr=None)

        N_FFT = self.getNearestLen(0.25, sr)

        y = self.normalizeVoiceLen(y, self.VOICE_LEN)  # 归一化长度

        mfcc_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=int(N_FFT / 4))
        feature = np.mean(mfcc_data, axis=0)
        feature = feature.reshape((126, 1))
        feature -= self.DATA_MEAN
        feature /= self.DATA_STD
        feature = feature.reshape((1, 126, 1))
        result = self.model.predict(feature)
        index = np.argmax(result, axis=1)[0]
        # print(self.edr[index])
        return self.edr[index]

    def get_wav_blocks(self,audio_file, num_seconds_each_file):
        # file_name = "haodf.mp3"

        # First load the file
        audio, sr = librosa.load(audio_file)

        # Get number of samples for 2 seconds; replace 2 by any number
        buffer = num_seconds_each_file * sr

        samples_total = len(audio)
        samples_wrote = 0
        counter = 1

        list_blocks=[]
        while samples_wrote < samples_total:

            # check if the buffer is not exceeding total samples
            if buffer > (samples_total - samples_wrote):
                buffer = samples_total - samples_wrote

            block = audio[samples_wrote: (samples_wrote + buffer)]
            list_blocks.append((block,sr))

            counter += 1
            samples_wrote += buffer
        return list_blocks

    def get_wav_files(self,audio_file, num_seconds_each_file, temp_folder):
        # file_name = "haodf.mp3"

        # First load the file
        audio, sr = librosa.load(audio_file)

        # Get number of samples for 2 seconds; replace 2 by any number
        buffer = num_seconds_each_file * sr

        samples_total = len(audio)
        samples_wrote = 0
        counter = 1
        list_files = []
        while samples_wrote < samples_total:

            # check if the buffer is not exceeding total samples
            if buffer > (samples_total - samples_wrote):
                buffer = samples_total - samples_wrote

            block = audio[samples_wrote: (samples_wrote + buffer)]
            out_filename = temp_folder + "/split_" + str(counter) + "_" + audio_file
            out_filename = out_filename.replace(".mp3", ".wav")
            list_files.append(out_filename)
            # Write 2 second segment
            # librosa.output.write_wav(out_filename, block, sr)
            sf.write(out_filename, block, sr, 'PCM_24')
            counter += 1
            samples_wrote += buffer
        return list_files

    def get_emotion_list(self,audio_file,num_sec_each_file,temp_folder):
        list_files = self.get_wav_files(audio_file, num_sec_each_file,temp_folder=temp_folder)
        list_ts=[]
        list_emo=[]
        count=0
        for file in list_files:
            emotion = self.analyze(file)
            list_emo.append(emotion)
            list_ts.append([count*num_sec_each_file,(count+1)*num_sec_each_file])
            os.remove(file)
            count+=1
        return list_emo,list_ts

    def get_emotion_list_by_blocks(self, audio_file, num_sec_each_file):
        list_blocks = self.get_wav_blocks(audio_file, num_sec_each_file)
        list_ts = []
        list_emo = []
        count = 0
        for block_info in list_blocks:
            emotion = self.analyze_block(block_info[0],block_info[1])
            list_emo.append(emotion)
            list_ts.append([count * num_sec_each_file, (count + 1) * num_sec_each_file])
            count += 1
        return list_emo,list_ts

def example_use():
    speech_kit = SpeechEmotionToolkit(model_path='speech_emotion/speech_mfcc_model.h5', model_para_path='speech_emotion/mfcc_model_para_dict.pkl')
    speech_kit.load()
    list_emo, list_timestamp = speech_kit.get_emotion_list_by_blocks(audio_file="speech_emotion/haodf.mp3", num_sec_each_file=5)

    print("Time interval\tEmotion")
    for idx, e in enumerate(list_emo):
        print(list_timestamp[idx], "\t", e)

# example_use()


