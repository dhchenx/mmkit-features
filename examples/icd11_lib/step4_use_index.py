from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib

mmf_file=f"icd11_mmf/icd11_rel.mmf"

print("loading mmf files...")
feature_lib=MMFeaturesLib(file_path=mmf_file)

print("creating plain text index...")
# feature_lib.to_index_file("text","icd11_mmf/index_test/text.index",index_type="brutal_force")

print("creating inverted index....")
# feature_lib.to_index_file("text","icd11_mmf/index_test/text_inverted.index",index_type="inverted_index")

print("creating positional text...")
# feature_lib.to_index_file("text","icd11_mmf/index_test/text_positional.index",index_type="positional_index")

print("searching plain index test....")
result_bf=feature_lib.search_index(index_file_path="icd11_mmf/index_test/text.index",query="Fungal infection",search_type="brutal_force")
print(result_bf)

print("searching inverted index test....")
result_bf=feature_lib.search_index(index_file_path="icd11_mmf/index_test/text_inverted.index",query="Fungal infection",search_type="inverted_index")
print(result_bf)

print("searching positional index test....")
result_bf=feature_lib.search_index(index_file_path="icd11_mmf/index_test/text_positional.index",query="Fungal infection",search_type="positional_index")
print(result_bf)