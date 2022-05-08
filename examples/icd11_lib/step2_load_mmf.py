from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib

flag="icd11"

mmf_file=f"icd11_mmf/{flag}.mmf"

feature_lib=MMFeaturesLib(file_path=mmf_file)

# data=feature_lib.get_data()

#for key in data.keys():
#    content=data[key]
#    print(content)

# feature_lib.show_structure()
feature_lib.show_sample_data(max_num=5)

