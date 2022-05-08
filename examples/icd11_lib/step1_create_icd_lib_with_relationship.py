import os
import sys
import time
import csv
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
icd11_lib.set_compressed(False)
# 1 Processing the nodes
# 1.1 read all ICD code data
print("starting read ICD node data...")
features_dict={}
with open(graph_nodes_file, mode='r',encoding='utf-8-sig') as csv_file:
    csv_reader = csv.DictReader(csv_file,delimiter=",",fieldnames=fields)
    line_count = 0
    # for each ICD code information
    for row in tqdm(csv_reader):
        # obtain ICD code as code
        id=row[":ID"]
        # set up fields values
        meta_fields = {
            "name": str(row["name"]),
            "text": str(row["description"]),
            "modality": "text",
            "feature_ids": [],
            "feature_dim_names": [],
            "feature_extractor": "mmk",
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
            if k not in [":ID",":LABEL","f_uri","name","description"]:
                list_other[k]=v
        meta_fields["attributes"]=list_other
        # setup a multimodal features node
        mmnode = MMFeaturesNode(content_id=id,meta_fields=meta_fields)

        # mmnode.set_item("attributes",{"split":list_split[id]})
        # add the node to a dictionary
        mmnode.validate_empty_field()
        #if icd11_lib.use_alias_names:
        #    mmnode.map_to_short_keys()
        features_dict[id] = mmnode

# 1. read index terms
print("starting ICD index term data...")
index_field_names=":ID,f_url,term,:LABEL".split(",")
index_file=root_path+"/index_nodes.txt"
with open(index_file, mode='r',encoding='utf-8-sig') as csv_file:
    csv_reader = csv.DictReader(csv_file,delimiter=",",fieldnames=index_field_names)
    line_count = 0
    # for each ICD code information
    for row in tqdm(csv_reader):
        id=row[":ID"]
        index_fields = {
            "name": "",
            "text": str(row["term"]),
            "modality": "text",
            "feature_ids": [],
            "feature_dim_names": [],
            "feature_extractor": "mmk",
            "features": np.array([[]]),
            "intervals": np.array([[]]),
            "space": [],
            "labels": str(row[":LABEL"]).split(";"),
            "origin": str(row["f_url"]),
            "attributes": [],
            "links": []
        }
        features_dict[id]=index_fields

# 2. start processing relationships
rels_dict={}
rel_id=0

# 2.1 read index relationships
index_rel_file=root_path+"/"+"graph_pcs_rels.txt"
index_rel_fields=[":START_ID","src",":END_ID",":TYPE"]
print("starting to read index relationships...")
with open(index_rel_file, mode='r',encoding='utf-8-sig') as csv_file:
    csv_reader = csv.DictReader(csv_file,delimiter=",",fieldnames=index_rel_fields)
    # for each ICD code information
    for row in tqdm(csv_reader):
        if row[":START_ID"]==":START_ID":
            continue
        start_id=row[":START_ID"]
        end_id=row[":END_ID"]
        rel_type=row[":TYPE"]
        rel={
            "start":start_id,
            "end":end_id,
            "type":rel_type,
            "direction":"uni",# uni, bi, none
            "function":"",
            "attributes":{}
        }
        attrs={}
        for k in row.keys():
            if k not in [':START_ID',':END_ID',":TYPE"]:
                attrs[k]=row[k]
        rel["attributes"]=attrs

        rels_dict[str(rel_id)]=rel
        rel_id+=1

# 2.2 read relationships
print("starting read node is-a relationship")
relationship_file=root_path+"/"+"graph_rels_import.txt"
relationship_fields=[":START_ID","role",":END_ID",":TYPE"]

with open(relationship_file, mode='r',encoding='utf-8-sig') as csv_file:
    csv_reader = csv.DictReader(csv_file,delimiter=",",fieldnames=relationship_fields)
    line_count = 0
    # for each ICD code information
    for row in tqdm(csv_reader):
        if row[":START_ID"]==":START_ID":
            continue
        start_id=row[":START_ID"]
        end_id=row[":END_ID"]
        rel_type=row[":TYPE"]
        rel={
            "start":start_id,
            "end":end_id,
            "type":rel_type,
            "direction":"uni",# uni, bi, none
            "function":"",
            "attributes":{
                "role":row["role"]
            }
        }
        rels_dict[str(rel_id)]=rel
        rel_id+=1

# 2.3 read post-coordination relationships
post_relationship_file=root_path+"/"+"graph_pcs_rels.txt"
print("starting to read post-coordination data...")
with open(post_relationship_file, mode='r',encoding='utf-8-sig') as csv_file:
    csv_reader = csv.DictReader(csv_file,delimiter=",")
    # for each ICD code information
    for row in tqdm(csv_reader):
        if row[":START_ID"]==":START_ID":
            continue
        start_id=row[":START_ID"]
        end_id=row[":END_ID"]
        rel_type=row[":TYPE"]
        rel={
            "start":start_id,
            "end":end_id,
            "type":rel_type,
            "direction":"uni",# uni, bi, none
            "function":"",
            "attributes":{}
        }
        attrs={}
        for k in row.keys():
            if k not in [':START_ID',':END_ID',":TYPE"]:
                attrs[k]=row[k]
        rel["attributes"]=attrs

        rels_dict[str(rel_id)]=rel
        rel_id+=1

# 5. set data to the above feature dictionary with content id key
print("set data...")
icd11_lib.set_data(features_dict)
# 5.1 set relationships data
icd11_lib.set_rel_data(rels_dict)
print("writing data...")
# 6. you can save the features to disk
icd11_lib.save_data(f"icd11_mmf/icd11_rel.mmf")
end_time=time.time()
print("time cost: ",end_time-start_time)
# 7. you can check structure of csd file which happens to be the format of h5dy
# icd11_lib.show_structure(f"medmnist_mmf/{flag}.mmf")
# 8. have a glance of features content within the dataset
# print("showing some sample data from the file...")
# icd11_lib.show_sample_data(f"icd11_mmf/icd11_rel.mmf")

