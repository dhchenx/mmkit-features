import pickle

list_id_text=pickle.load(open("icd11_mmf/index/index_id_text.pickle",'rb'))

queries = ['gram-negative bacteria', 'Fungal infection', 'purulent exudate']

import time
start_time=time.time()
for query in queries:
    list_ids=[]
    for item in list_id_text:
        text=item[1]
        if query in text:
            list_ids.append(item[0])
    # print(list_ids)

end_time=time.time()
time_passed = round((end_time - start_time)*1000,2)
print("time:", time_passed,"ms")

