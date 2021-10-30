from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
from tqdm import tqdm
import pickle
flag="icd11"

mmf_file=f"icd11_mmf/{flag}_rel.mmf"

feature_lib=MMFeaturesLib(file_path=mmf_file)

data=feature_lib.get_data()

list_id_text=[]
list_id_name=[]
for key in tqdm(data.keys()):
    content=data[key]
    content_id=key
    name=content["name"][()]
    if 'text' in content.keys():
        text=content["text"][()]
    else:
        text=''
    list_id_text.append([key,text])
    list_id_name.append([key,name])
    # print(content_id,name)

pickle.dump(list_id_text,open("icd11_mmf/index/index_id_text.pickle",'wb'))
pickle.dump(list_id_name,open("icd11_mmf/index/index_id_name.pickle",'wb'))




