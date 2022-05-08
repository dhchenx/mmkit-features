from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
import time

mmf_file=f"../datasets/birds.mmf"
list_result=[]
start_time=time.time()

print("loading mmf files...")
birds_lib=MMFeaturesLib(file_path=mmf_file)

# creating inverted index
birds_lib.to_index_file("text","../datasets/text_inverted.index",index_type="inverted_index")
time_inverted=time.time()-start_time
list_result.append(("inverted indexing",time_inverted))

print("time cost of creating inverted: ",time.time()-start_time)

# start to perform search test
query_str="large brown wings"

print("searching inverted index test....")
start_time=time.time()
result_bf=birds_lib.search_index(index_file_path="../datasets/text_inverted.index",query=query_str,search_type="inverted_index")
print(result_bf)

print("time cost of search inverted: ",time.time()-start_time)

num_match=0
for key in result_bf:
    content=birds_lib.get_content_by_id(key)
    text=content["text"][()].decode("utf-8","ignore")
    print(key,text)
    if query_str in text:
        num_match+=1
print("p = ",round(num_match*1.0/len(result_bf),4))
