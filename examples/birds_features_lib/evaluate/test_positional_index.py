from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
import time

mmf_file=f"../datasets/birds.mmf"
list_result=[]
start_time=time.time()

print("loading mmf files...")
birds_lib=MMFeaturesLib(file_path=mmf_file)
print("loading ",time.time()-start_time)
# creating inverted index
print("creating positional text...")
start_time=time.time()
birds_lib.to_index_file("text","../datasets/text_positional.index",index_type="positional_index")

time_positional=time.time()-start_time
list_result.append(("positional indexing",time_positional))

# start to perform search test
query_str="large brown wings"


print("searching positional index test....")
start_time=time.time()
result_bf=birds_lib.search_index(index_file_path="../datasets/text_positional.index",query=query_str,search_type="positional_index")
print(result_bf)

search_time_positional=time.time()-start_time

print("search time: ",search_time_positional)

# show results
num_match=0
for key in result_bf:
    content=birds_lib.get_content_by_id(key)
    text=content["text"][()].decode("utf-8","ignore")
    print(key,text)
    if query_str in text:
        num_match+=1

print("p = ",round(num_match*1.0/len(result_bf),4))

# export key values

# birds_lib.export_key_values("text",save_path="text.csv")

