## Audio Features Extraction

This module aims to provide a framework to generate audio features from an audio file like `wav` file so the audio features can be used in the future. 

The audio file has several characteristics such as volume, ZeroCR, and MFCC. 

Here is an example to show how to use the `audio` module. 

```python
from mmkfeatures.speech.audio_features_wrapper import AudioFeaturesWrapper

if __name__ == '__main__':
    audio_feat_wrapper=AudioFeaturesWrapper()
    sound_file="../data/sounds/english.wav"
    
    # 1. get MFCC features
    features1=audio_feat_wrapper.get_mfcc_features2(sound_file)
    print(features1)

    # audio_feat_wrapper.show_plot(sound_file)
    # audio_feat_wrapper.show_features(sound_file)
    # audio_feat_wrapper.show_scaled_features(sound_file)

    # 2. get other features like volume, ...
    # volume
    # fs_volume=audio_feat_wrapper.get_volume_features(sound_file)
    # print(fs_volume[2])

    # 3. get ZeroCR features
    # ZeroCR
    fs_zerocr=audio_feat_wrapper.get_volume_features(sound_file)
    print(fs_zerocr)
```