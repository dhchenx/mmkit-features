from mmkfeatures.speech.audio_features_wrapper import AudioFeaturesWrapper

if __name__ == '__main__':
    audio_feat_wrapper=AudioFeaturesWrapper()
    sound_file="../data/sounds/english.wav"
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

