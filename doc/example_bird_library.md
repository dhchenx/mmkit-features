## A tutorial to create a multimodal features library using CUB_200_2011 datasets. 

Here we give a step-by-step example to demonstrate the steps to build a multimodal feature library using the CUB_200_2011 datasets. The CUB_200_2011 dataset contains text and image-based content to illustrate the classification of birds all over the world. The number of bird categories is 200. The database contains rich multimodal information about birds. 

### Steps

1. Download your CUB_200_2011 datasets and store them into one single folder. 

The dataset folder should contains text descriptions and images of birds. 

2. Set up a new empty multimodal features library by setting the root_name and dataset_name during init(). 

3. Define the necessary file path list for bird data's text and image. 

4. Load multimodal dictionaries

5. Load all bird images grouped by their categories

6. Load all part image from a bird's part location and crop the available image from the original image using the bounding box size. Each multimodal node contains all parts information including their labels. 

7. Save the database. 

### Code example

```python
from tqdm import tqdm
import cv2
import numpy as np
from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
from mmkfeatures.fusion.mm_features_node import MMFeaturesNode
from mmkfeatures.image.color_descriptor import ColorDescriptor
from mmkfeatures.image.image_processing import ImageUtils

root_path=r"datasets/CUB_200_2011"

img_utils=ImageUtils()

# set up multimodal features library
birds_lib=MMFeaturesLib(root_name="CUB",dataset_name=f"CUB-200-2011",
                        description="This is a multimodal bird database!")
# compress the unnecessary fields
birds_lib.set_compressed(False)
# read all ICD code data
features_dict={}

images_list_file=root_path+"/images.txt"

class_list_file=root_path+"/classes.txt"

image_class_list_file=root_path+"/image_class_labels.txt"

bounding_list=root_path+"/bounding_boxes.txt"

part_list_file=root_path+"/parts/part_locs.txt"

part_name_list_file=root_path+"/parts/parts.txt"

bird_dicts={}

dict_class_names={}
for item in open(class_list_file,"r",encoding='utf-8').read().split("\n"):
    if item.strip()=="":
        continue
    fs=item.split(" ")
    dict_class_names[fs[0]]=fs[1]

bird_dicts["class_names"]=dict_class_names

dict_image_class={}
for item in open(image_class_list_file,"r",encoding='utf-8').read().split("\n"):
    if item.strip()=="":
        continue
    fs=item.split(" ")
    dict_image_class[fs[0]]=fs[1]

dict_bounding={}
for item in open(bounding_list,"r",encoding='utf-8').read().split("\n"):
    if item.strip()=="":
        continue
    fs=item.split(" ")
    dict_bounding[fs[0]]=[int(float(fs[1])),int(float(fs[2])),int(float(fs[3])),int(float(fs[4]))]

dict_parts={}
for item in open(part_list_file,"r",encoding='utf-8').read().split("\n"):
    if item.strip()=="":
        continue
    fs=item.split(" ")
    if fs[0] not in dict_parts.keys():
        dict_parts[fs[0]]=[]
    dict_parts[fs[0]].append([int(float(fs[1])),int(float(fs[2])),int(float(fs[3])),int(float(fs[4]))])

dict_part_names={}
for item in open(part_name_list_file,"r",encoding='utf-8').read().split("\n"):
    if item.strip()=="":
        continue
    fs=item.split(" ")
    dict_part_names[fs[0]]=fs[1]

bird_dicts["part_names"]=dict_part_names

birds_lib.set_dicts(bird_dicts)

cd = ColorDescriptor((8, 12, 3))

for item in tqdm(open(images_list_file,"r",encoding='utf-8').read().split("\n")):
    if item.strip()=="":
        continue
    fs=item.split(" ")
    image_id=fs[0]
    image_path=root_path+"/images/"+fs[1]
    # text
    text_path=root_path+"/text_c10/"+fs[1].replace(".jpg",".txt")
    text=open(text_path,"r",encoding='utf-8').read()
    # print(text)
    # print(image_path)
    # 1. hist features
    # print(image_id)
    # print(image_path)
    image = cv2.imread(image_path)
    # print("image shape: ",image.shape)
    bbox=dict_bounding[image_id]
    # print(bbox)
    box_image = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

    image_features=cd.describe(box_image)
    # 2. raw numpy data
    # image_features=img_utils.get_numpy_array_rgb(image_path)
    feature_ids=[]
    locations=[]

    features=[]
    # parts
    items = dict_parts[image_id]
    for item in items:
        kid = item[0]
        x = item[1]
        y = item[2]
        visible = item[3]
        feature_ids.append(kid)
        locations.append([x, y, visible])
        if int(visible)==1:
            point = (x, y)
            width = 50
            y1 = point[1] - width
            y2 = point[1] + width
            x1 = point[0] - width
            x2 = point[0] + width
            if y1 < 0:
                y1 = 1
            if y2 > image.shape[0]:
                y2 = image.shape[0]
            if x1 < 0:
                x1 = 1
            if x2 > image.shape[1]:
                x2 = image.shape[1]
            # print("point:",point)
            # print(x1,x2,y1,y2)
            part_image = image[y1:y2, x1:x2]
            features.append(cd.describe(part_image))
            # print(len(cd.describe(part_image)))
        else:
            features.append([0 for _ in range(1440)])
        # print()

    meta_fields = {
        "name": "",
        "text": text,
        "modality": "text,image",
        "raw":np.array([image_features]),
        "feature_dim_names": [],
        "feature_extractor": "mmkit",
        "feature_ids": feature_ids,
        "features": np.array(features),
        "intervals": np.array([[]]),
        "locations": locations,
        "labels": [dict_image_class[image_id]],
        "origin": fs[1],
        "attributes": {
            "bounding box": dict_bounding[image_id]
        },
        "links": []
    }
    mmnode = MMFeaturesNode(content_id=image_id, meta_fields=meta_fields)

    mmnode.validate_empty_field()

    features_dict[image_id]=mmnode

    # print()

birds_lib.set_data(features_dict)
# 6. you can save the features to disk
birds_lib.save_data(f"datasets/birds.mmf")

```

### Load MMF libraries

```python
from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib

mmf_file=f"datasets/birds.mmf"

feature_lib=MMFeaturesLib(file_path=mmf_file)

data=feature_lib.get_data()

feature_lib.show_sample_data(max_num=5)
```

### Use text indexing

```python
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


```

### Create and Use image indexing

```python
from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
from mmkfeatures.image.color_descriptor import ColorDescriptor
import cv2

# load an existing multimodal feature lib
mmf_file=f"datasets/birds_raw.mmf"
feature_lib=MMFeaturesLib(file_path=mmf_file)
data=feature_lib.get_data()

# set test image and index file's path
test_img_path="datasets/CUB_200_2011/images/005.Crested_Auklet/Crested_Auklet_0001_794941.jpg"
index_path="datasets/image.index"

# create index
feature_lib.to_obj_index(index_file=index_path,obj_field="objects",index_type="color_descriptor")

# query index by color_descriptor
cd = ColorDescriptor((8, 12, 3))
query_image = cv2.imread(test_img_path)
query_features = cd.describe(query_image)
search_results=feature_lib.search_obj_index(index_file=index_path,features=query_features)

# loop over the results
for (score, resultID) in search_results:
    print(resultID,score)
    content=feature_lib.get_content_by_id(resultID)
    print(content)
    image=content["objects"]["0"][()]
    title=content["labels"][()][0]
    cv2.imshow(title, image)
    cv2.waitKey(0)
```

