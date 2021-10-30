from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib

flag="pneumoniamnist"

mmf_file=f"medmnist_mmf/{flag}.mmf"

feature_lib=MMFeaturesLib(file_path=mmf_file)

print(feature_lib.meta_dict)

# feature_lib.show_structure()
feature_lib.show_sample_data(max_num=2)

