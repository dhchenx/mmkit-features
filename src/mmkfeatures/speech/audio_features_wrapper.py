import os
from mmkfeatures.speech.MFCC import *

import scipy.io.wavfile as wav
import mmkfeatures.speech.feature_extraction_functions
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sklearn
import wave
import pylab as pl
import numpy as np
import mmkfeatures.speech.Volume as vp
import mmkfeatures.speech.ZeroCR

class AudioFeaturesWrapper:
    def __init__(self):
        pass

    def get_mfcc_features(self,sound_file):
        # "../sounds/english.wav"
        (rate, sig) = wav.read(sound_file)
        mfcc_feat = mfcc(sig, rate)
        d_mfcc_feat = delta(mfcc_feat, 2)
        fbank_feat = logfbank(sig, rate)
        print(fbank_feat[1:3, :])
        return fbank_feat[1:3, :]

    def get_mfcc_features2(self,sound_file,n_mfcc=24):
        sig, rate = librosa.load(sound_file)
        # (rate, sig) = wav.read(sound_file)
        # sig = sig[0:int(3.5 * rate)]
        features=mmkfeatures.speech.feature_extraction_functions.get_feature(rate,sig,n_mfcc=n_mfcc)
        return features

    def show_features(self,sound_file):
        x, fs = librosa.load(sound_file)
        # librosa.display.waveplot(x, sr=fs)
        mfccs = librosa.feature.mfcc(x, sr=fs)
        librosa.display.specshow(mfccs, sr=fs, x_axis='time')
        plt.show()

    def show_plot(self,sound_file):
        x, fs = librosa.load(sound_file)
        librosa.display.waveplot(x, sr=fs)
        plt.show()

    def show_scaled_features(self,sound_file):
        x, fs = librosa.load(sound_file)
        # librosa.display.waveplot(x, sr=fs)
        mfccs = librosa.feature.mfcc(x, sr=fs)
        mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
        print(mfccs.mean(axis=1))
        print(mfccs.var(axis=1))
        librosa.display.specshow(mfccs, sr=fs, x_axis='time')
        plt.show()

    def get_volume_features(self,sound_file):
        # read wave file and get parameters.
        fw = wave.open(sound_file, 'rb')
        params = fw.getparams()
        print(params)
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = fw.readframes(nframes)
        waveData = np.fromstring(strData, dtype=np.int16)
        waveData = waveData * 1.0 / max(abs(waveData))  # normalization
        fw.close()

        # calculate volume
        frameSize = 256
        overLap = 128
        volume11 = vp.calVolume(waveData, frameSize, overLap)
        volume12 = vp.calVolumeDB(waveData, frameSize, overLap)
        return [waveData,volume11,volume12] # amplitude, absSum, Decibel(dB)

    def get_zero_cr_features(self,sound_file):
        # read wave file and get parameters.
        fw = wave.open('../sounds/aeiou.wav', 'rb')
        params = fw.getparams()
        print(params)
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = fw.readframes(nframes)
        waveData = np.fromstring(strData, dtype=np.int16)
        waveData = waveData * 1.0 / max(abs(waveData))  # normalization
        fw.close()

        # calculate Zero Cross Rate
        frameSize = 256
        overLap = 0
        zcr = AudioFeatures.ZeroCR.ZeroCR(waveData, frameSize, overLap)
        return zcr

if __name__ == '__main__':
    audio_feat_wrapper=AudioFeaturesWrapper()
    sound_file="../dataset/sounds/english.wav"
    # features=audio_feat_wrapper.get_mfcc_features(sound_file)
    # print(features)
    # print("-=======================")
    features1=audio_feat_wrapper.get_mfcc_features2(sound_file)
    print(features1)
    # audio_feat_wrapper.show_plot(sound_file)
    # audio_feat_wrapper.show_features(sound_file)
    # audio_feat_wrapper.show_scaled_features(sound_file)

    # volume
    # fs_volume=audio_feat_wrapper.get_volume_features(sound_file)
    # print(fs_volume[2])
    # ZeroCR
    fs_zerocr=audio_feat_wrapper.get_volume_features(sound_file)
    print(fs_zerocr)


