from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib

mmf_file=f"datasets/birds.mmf"

print("loading mmf files...")
birds_lib=MMFeaturesLib(file_path=mmf_file)

print("creating plain text index...")
birds_lib.to_index_file("text","datasets/text.index",index_type="brutal_force")

print("creating inverted index....")
birds_lib.to_index_file("text","datasets/text_inverted.index",index_type="inverted_index")

print("creating positional text...")
birds_lib.to_index_file("text","datasets/text_positional.index",index_type="positional_index")

# start to perform search test
query_str="yellow breast"

print("searching plain index test....")
result_bf=birds_lib.search_index(index_file_path="datasets/text.index",query=query_str,search_type="brutal_force")
print(result_bf)

print("searching inverted index test....")
result_bf=birds_lib.search_index(index_file_path="datasets/text_inverted.index",query=query_str,search_type="inverted_index")
print(result_bf)

print("searching positional index test....")
result_bf=birds_lib.search_index(index_file_path="datasets/text_positional.index",query=query_str,search_type="positional_index")
print(result_bf)

