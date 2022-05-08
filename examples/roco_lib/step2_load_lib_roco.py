from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib


mmf_file=f"datasets/roco.mmf"

feature_lib=MMFeaturesLib(file_path=mmf_file)

data=feature_lib.get_data()

feature_lib.show_sample_data(max_num=5)

