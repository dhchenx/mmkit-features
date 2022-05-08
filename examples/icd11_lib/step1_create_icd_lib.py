import os
import sys
import csv
import time

from tqdm import tqdm
import numpy as np
from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
from mmkfeatures.fusion.mm_features_node import MMFeaturesNode
start_time=time.time()
root_path=r"D:\GitHub\ICD11-CAC\ICD-11-graph-databases\english_full"

graph_nodes_file=root_path+"/graph_nodes_import.txt"

fields=":ID,code,name,:LABEL,f_uri,l_uri,classkind,depthinkind:int," \
       "isresidual,primarylocation,chapterno,isleaf,anypostcoordination," \
       "chinese_name,foundation_uri,description,codingnotes".split(',')

# set up multimodal features library
icd11_lib=MMFeaturesLib(root_name="ICD-11",dataset_name=f"ICD-11-Features",
                        description="This is a test ICD-11 knowledge base!")
# compress the unnecessary fields
# icd11_lib.set_compressed(True)
# read all ICD code data
features_dict={}
with open(graph_nodes_file, mode='r',encoding='utf-8-sig') as csv_file:
    csv_reader = csv.DictReader(csv_file,delimiter=",",fieldnames=fields)
    line_count = 0
    # for each ICD code information
    for row in tqdm(csv_reader):
        # obtain ICD code as code
        code=row["code"]
        # set up fields values
        meta_fields = {
            "name": str(row["name"]),
            "text": str(row["description"]),
            "modality": "text",
            "feature_ids": [str(row[":ID"])],
            "feature_dim_names": [],
            "feature_extractor": "mmkit-features",
            "features": np.array([[]]),
            "intervals": np.array([[]]),
            "space": [],
            "labels": str(row[":LABEL"]).split(";"),
            "origin": str(row["f_uri"]),
            "attributes":[] ,
            "links": []
        }

        # save other ICD-11 attributes to attributes
        list_other= {}
        for k in row.keys():
            v=row[k]
            if k not in ["code",":ID",":LABEL","f_uri","name","description"]:
                list_other[k]=v
        meta_fields["attributes"]=list_other
        # setup a multimodal features node
        mmnode = MMFeaturesNode(content_id=code,meta_fields=meta_fields)

        # mmnode.set_item("attributes",{"split":list_split[id]})
        # add the node to a dictionary
        mmnode.validate_empty_field()
        #if icd11_lib.use_alias_names:
        #    mmnode.map_to_short_keys()
        features_dict[code] = mmnode

# 5. set data to the above feature dictionary with content id key
icd11_lib.set_data(features_dict)
# 6. you can save the features to disk
icd11_lib.save_data(f"icd11_mmf/icd11.mmf")
# 7. you can check structure of csd file which happens to be the format of h5dy
# icd11_lib.show_structure(f"medmnist_mmf/{flag}.mmf")
# 8. have a glance of features content within the dataset
# icd11_lib.show_sample_data(f"icd11_mmf/icd11.mmf")

end_time=time.time()
print("time cost: ",end_time-start_time)

