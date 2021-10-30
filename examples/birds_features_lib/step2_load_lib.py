from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib


mmf_file=f"datasets/birds.mmf"

feature_lib=MMFeaturesLib(file_path=mmf_file)

data=feature_lib.get_data()

#for key in data.keys():
#    content=data[key]
#    print(key)

# feature_lib.show_structure()
feature_lib.show_sample_data(max_num=5)

