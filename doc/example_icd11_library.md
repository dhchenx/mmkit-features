## Establishing ICD-11 disease coding library

This example demonstrates steps to create a multimodal feature library using the datasets from International Classification of Diseases, Eleventh Revision (ICD-11). The ICD-11 datasets contains massive text description of disease entities and their complicated relationships. It is also suitable for use to show the use of the `mmkit-features` toolkit. 

### Steps 

1. Prepare the ICD-11 datasets

The datasets contain ICD-11 entity information and their relationships. 

2. Set up a MMF library

3. Read ICD-11 entity files

4. Read ICD-11 index term files

5. Read index term relationships file

6. Read node relationships file

7. Read ICD-11's post-coordination relationships file

8. Set the data inside the library and save the MMF library. 

### Code example

A full example of creating the MMF library with the use of relationships is below:

```python
import os
import sys
import csv
from tqdm import tqdm
import numpy as np
from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
from mmkfeatures.fusion.mm_features_node import MMFeaturesNode

root_path=r"ICD-11-graph-databases\english_full"

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
# 7. you can check structure of csd file which happens to be the format of h5dy
# icd11_lib.show_structure(f"medmnist_mmf/{flag}.mmf")
# 8. have a glance of features content within the dataset
print("showing some sample data from the file...")
icd11_lib.show_sample_data(f"icd11_mmf/icd11_rel.mmf")


```

### Load MMF library

```python
from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib

flag="icd11"

mmf_file=f"icd11_mmf/{flag}_rel.mmf"

feature_lib=MMFeaturesLib(file_path=mmf_file)

feature_lib.show_sample_data(max_num=5)
```

### Export indexing files and search indexing files

```python
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
```

