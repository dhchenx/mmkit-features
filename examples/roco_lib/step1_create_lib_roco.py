import os
import sys
import csv
from tqdm import tqdm
import cv2
import numpy as np
import time
import pickle
from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
from mmkfeatures.fusion.mm_features_node import MMFeaturesNode
from mmkfeatures.image.color_descriptor import ColorDescriptor
from mmkfeatures.image.image_processing import ImageUtils
start_time=time.time()
image_root_path=r"datasets/all_data"
data_root_path="datasets/radiology"

img_utils=ImageUtils()

# set up multimodal features library
roco_lib=MMFeaturesLib(root_name="ROCO",dataset_name=f"ROCO-DATASET",
                        description="This is a multimodal roco radiology image database!")
# compress the unnecessary fields
roco_lib.set_compressed(False)
# read all content data
features_dict={}

list_image_file=pickle.load(open(data_root_path+"/list_image_file.pickle","rb"))
list_image_caption=pickle.load(open(data_root_path+"/list_image_caption.pickle","rb"))
list_image_keywords=pickle.load(open(data_root_path+"/list_image_keywords.pickle","rb"))
list_image_lic=pickle.load(open(data_root_path+"/list_image_lic.pickle","rb"))
list_image_link=pickle.load(open(data_root_path+"/list_image_link.pickle","rb"))
list_image_semtypes=pickle.load(open(data_root_path+"/list_image_semtypes.pickle","rb"))
list_image_text=pickle.load(open(data_root_path+"/list_image_text.pickle","rb"))
list_class_name=pickle.load(open(data_root_path+"/list_class_name.pickle","rb"))
list_class_tag=pickle.load(open(data_root_path+"/list_class_tag.pickle","rb"))
list_image_cuis=pickle.load(open(data_root_path+"/list_image_cuis.pickle","rb"))

cd = ColorDescriptor((8, 12, 3))

for roco_id in tqdm(list(list_image_file.keys())):
    item=list_image_file[roco_id]

    file_path=image_root_path+f"/{item[0]}/radiology/images/{item[1]}"
    file_name=item[1]
    flag=item[0]
    caption=""
    lic=""
    link=""
    link_name=""
    semtypes=[]
    keywords=[]
    cuis=[]
    if roco_id in list_image_caption:
        caption=list_image_caption[roco_id]
    if roco_id in list_image_link:
        link=list_image_link[roco_id][0]
        link_name = list_image_link[roco_id][1]
    if roco_id in list_image_semtypes:
        semtypes=list_image_semtypes[roco_id]
    if roco_id in list_image_lic:
        lic=list_image_lic[roco_id]
    if roco_id in  list_image_keywords:
        keywords=list_image_keywords[roco_id]
    if roco_id in list_image_cuis:
        cuis=list_image_cuis[roco_id]

    if os.path.exists(file_path):
        # print(file_path)

        try:
            image = cv2.imread(file_path, 0)
            image_shape = image.shape
            if len(image_shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            image_features = cd.describe(image)
            # image=cv2.resize(image,dsize=(64,64))
            # del image
        except:
            print("Error, Not Found: ", file_path)
            image = np.array([])
            image_features = []

    else:
        print("Not Found: ",file_path)
        image=np.array([])
        image_features=[]

    meta_fields = {
        # "name": file_name,
        "text": caption,
        "modality": "text,image",
        "raw": np.array([image]),
        "feature_dim_names": [],
        "feature_extractor": "mmkit",
        "feature_ids": [],
        "features": np.array([image_features]),
        "intervals": np.array([]),
        "locations": [],
        # "labels": list_image_semtypes,
        # "origin": link,
        "attributes": {
            # "cc": lic,
            # "keywords":keywords,
            # "link":link,
            # "link_name":link_name,
            # "cuis":cuis
        },
        "links": []
    }

    mmnode = MMFeaturesNode(content_id=roco_id, meta_fields=meta_fields)

    mmnode.validate_empty_field()

    features_dict[roco_id] = mmnode

roco_lib.set_data(features_dict)
# 6. you can save the features to disk
roco_lib.save_data(f"datasets/roco_text_image_raw_only.mmf")

end_time=time.time()
time_cost=end_time-start_time

print("time cost: ",time_cost)