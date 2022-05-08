from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
import time

mmf_file=f"../datasets/birds.mmf"

def get_exact_match_ratio(lib,result):
    num_match = 0
    for key in result:
        content = lib.get_content_by_id(key)
        text = content["text"][()].decode("utf-8", "ignore")
        # print(key, text)
        if query_str in text:
            num_match += 1
    print("p = ", round(num_match * 1.0 / len(result), 4))
    return  round(num_match * 1.0 / len(result), 4)

list_result=[]
start_time=time.time()


print("loading mmf files...")
birds_lib=MMFeaturesLib(file_path=mmf_file)

time_load=time.time()-start_time
list_result.append(("load",time_load))

print("creating plain text index...")
start_time=time.time()
birds_lib.to_index_file("text","../datasets/text.index",index_type="brutal_force")
time_brutal_force=time.time()-start_time


list_result.append(("brutal force indexing",time_brutal_force))

print("creating inverted index....")
start_time=time.time()
birds_lib.to_index_file("text","../datasets/text_inverted.index",index_type="inverted_index")
time_inverted=time.time()-start_time
list_result.append(("inverted indexing",time_inverted))

print("creating positional text...")
start_time=time.time()
birds_lib.to_index_file("text","../datasets/text_positional.index",index_type="positional_index")
time_positional=time.time()-start_time
list_result.append(("positional indexing",time_positional))

# start to perform search test
query_str="large brown wings"

print("searching plain index test....")
start_time=time.time()
result_bf=birds_lib.search_index(index_file_path="../datasets/text.index",query=query_str,search_type="brutal_force")
print(result_bf)
search_time_brutal=time.time()-start_time
list_result.append(("brutal force searching",search_time_brutal,get_exact_match_ratio(birds_lib,result_bf)))

print("searching inverted index test....")
start_time=time.time()
result_bf=birds_lib.search_index(index_file_path="../datasets/text_inverted.index",query=query_str,search_type="inverted_index")
print(result_bf)
search_time_inverted=time.time()-start_time
list_result.append(("inverted index searching",search_time_inverted,get_exact_match_ratio(birds_lib,result_bf)))

print("searching positional index test....")
start_time=time.time()
result_bf=birds_lib.search_index(index_file_path="../datasets/text_positional.index",query=query_str,search_type="positional_index")
print(result_bf)
search_time_positional=time.time()-start_time
list_result.append(("positional index searching",search_time_positional,get_exact_match_ratio(birds_lib,result_bf)))
print()
print("Indexing method\tTime cost\tExact match ratio")
for result in list_result:
    if len(result)==3:
        print(f"{result[0]}\t{result[1]}\t{result[2]}")

print()
print("Loading method\tTime cost")
for result in list_result:
    if len(result)==2:
        print(f"{result[0]}\t{result[1]}")


