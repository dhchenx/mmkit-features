## Multimodal Features Library

This section presents technical details of the multimodal features library provided by our ```mmkit-features``` toolkit. All features extraction and basic modules are used to support this module, which is our goal in the project. 

We aim to develop a common framework of multimodal feature library to extract, represent, store, fuse, retrieve multimodal features in a rapid and efficient way. With the use of the `MMFeaturesLib` object in the toolkit, we are easy to create a multimodal feature library.

### A toy example of multimodal features library

Let us write a simple toy multimodal features library as follows. 

A summary of the steps are below: 

1. Create an empty multimodal features library with root_name and dataset_names;
2. Set names for each dimension name for convenience;
3. Set a  list of content ids;
4. According to ids, assign a group of features with interval to corresponding content id;
5. Set the library's data;
6. Save the features to disk for future use;
7. Check structure of lib file with the format of h5py;
8. Have a glance of feature contents within the dataset. 

A complete Python example is here:

```python
from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
from mmkfeatures.fusion.mm_features_node import MMFeaturesNode
import numpy as np
if __name__ == "__main__":
    # 1. create an empty multimodal features library with root_name and dataset_names
    feature_lib=MMFeaturesLib(root_name="test features",dataset_name="test_features")
    # 2. set names for each dimension name for convenience
    feature_lib.set_features_name(["feature1","feature2","feature3"])
    # 3. set a  list of content ids
    content_ids=["content1","content2","content3"]
    # 4. according to ids, assign a group of features with interval to corresponding content id
    features_dict={}
    for id in content_ids:
        mmf_node=MMFeaturesNode(id)
        mmf_node.set_item("name",str(id))
        mmf_node.set_item("features",np.array([[1,2,3]]))
        mmf_node.set_item("intervals",np.array([[0,1]]))
        features_dict[id]=mmf_node
    # 5. set the library's data
    feature_lib.set_data(features_dict)
    # 6. save the features to disk for future use
    feature_lib.save_data("test6_feature.csd")
    # 7. check structure of lib file with the format of h5py
    feature_lib.show_structure("test6_feature.csd")
    # 8. have a glance of features content within the dataset
    feature_lib.show_sample_data("test6_feature.csd")
```

As you can see, the example used two key objects, namely, MMFFeaturesLib and MMFeaturesNode. The libray is made of massive multimodal nodes. Therefore, a content in previous examples can be considered as a multimodal node that stores all features in time order. For example, a 60-second video clip can be represented as a multimodal object in the multimodal library. 

### Structure of multimodal features library

The multimodal feature library (`MMFeaturesLib`) contains several types of content, namely, `metadata`, `data`, `rel_data`, `dicts`, and other basic library information. 

#### Basic information

We create a new library by specifying two basic parameters, namely, `root_name`, and `dataset_name`. the `root_name` is used inside h5py file content of the library and the `dataset_name` is used to represent a common and readable name for the library for specific applications. 

#### Meta data

The `metadata` object is a dictionary to store extended fields of the library. The dictionary is defined as below: 

```python
self.meta_dict = {
                "root name": self.root_name,  # name of the featureset
                "computational sequence description": description,  # name of the featureset
                "dimension names": None,
                "computational sequence version": "1.0",  # the version of featureset
                "alignment compatible": "",  # name of the featureset
                "dataset name": self.name,  # featureset belongs to which dataset
                "dataset version": version,  # the version of dataset
                "creator": creator,  # the author of the featureset
                "contact": contact,  # the contact of the featureset
                "featureset bib citation": "",  # citation for the paper related to this featureset
                "dataset bib citation": "",  # citation for the dataset,
                "attributes_alias":"",
                "attributes":""
            }
```

The `metadata` object is deprived of the original `computational_sequence` object from the CMU-Multimodal-SDK but we extend the object by adding more metadata to support complicated operations of the library, such as `attributes` and `attributes-alias`. 

#### Multimodal Node List

The `data` object in the library is a list of multimodal node in a time order. 

The implemented function `set_data` is below:
```python
    def set_data(self,list_content):

        if self.use_alias_names:
            self.compseq_data = {}
            ks=list(list_content.keys())
            for k in ks:
                self.compseq_data[k]=self.compress_content(list_content[k])
        else:
            self.compseq_data = list_content

        # self.compseq_data = list_content
        self.compseq.setData(self.compseq_data, self.name)
```

Here are properties/attributes of multimodal nodes:

```python
    '''
    Multimodal Features Node Structure

    Content Id:
        name: a short name for the content
        format: '', a string indicating which formatio of the content
        text: '', a long text to describe the content
        modality: '', a string indicate which modality of the content
        feature_ids: [], ids for each features, if [] then using feature index in link representation
        feature_dim_names: [[]], if empty then using using index as dim name
        feature_extractor: a registered name to extract features from the original file
        features: an numpy array [[]]
        intervals: [[]] starting time and ending time, where [] indicates not specific
        space: [length, width,height], default value [] represents not specific
        labels: a list of labels, useful for multi-label or binary classification;a single label represent all feature has the same label
        origin: "" the original content file path, optional
        attributes: dict{}, user-defined key-value properties
        links: a list of relationships between the content's features, between inside and outside content
            e.g. link format: (feat_1,(rel_direction,rel_type,rel_func),feat_2),
                    where feat_1,feat_2 can be feature ids outside current content using format: content:feat1
    '''
```

We can also use alias names for the attribute labels like below:
```python
    def get_all_validated_keys_alias(self):
        return ["ID", 'NA', 'TX', 'MD', 'FID', 'FDM',
                'F', 'FET', 'ITV', 'SP', 'LB', 'LBN',
                'OG', 'AT', 'LK', "FM","RAW","LOC","OBJ"
                ]
```

Below is an example of showing how to define a multimodal node in actual codes:
```python
...
meta_fields = {
        "name": "",
        "text": text,
        "modality": "text,image",
        "objects":[entire_image],
        "raw":raw_features,
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
...
```

#### Multimodal Relationships 

Since there are multimodal nodes in the library, we introduce the concept of multimodal relationships into the libarry. The object structure to represent relataionships between multimodal nodes is actual type of Python dictionary. 

Here is an example of defining relationships between multimodal nodes: 

```python
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
```

The `rel` object in the example contains attributes like `start`, `end`, `type`, `direction`, `function` and `attributes`. 

`start` represents the starting point of the relationships; otherwise, using `end`. 

`type` represents types of relationship, such bi-directional, unique directional and non-directional. 

`function` will be used in the future for integration with deep-learning based models to automate reasoning between nodes via relationships. 

`attributes` is used to store other user-related information related to the relationships between nodes. 

After we construct a `rel_data` dictionary, we can save the relationship data to library by:

```python
# 5.1 set relationships data
icd11_lib.set_rel_data(rels_dict)
```

#### Multimodal Dictionary

The multimodal dictionary is used as a dictionary to store some mapping values. For example, for labelled data, if you do not want to repeatedly use label name to represent their features' label, you can define a label-index dictionary and then use label index in the features. This will save storing spaces. 

Here is a example of defining a multimodal dictionary here:

```python
# create a multimodal features library
birds_lib=MMFeaturesLib(root_name="CUB",dataset_name=f"CUB-200-2011",
                        description="This is a multimodal bird database!")

# read class's label file
dict_class_names={}
for item in open(class_list_file,"r",encoding='utf-8').read().split("\n"):
    if item.strip()=="":
        continue
    fs=item.split(" ")
    dict_class_names[fs[0]]=fs[1]

# set dict name for the class list
bird_dicts["class_names"]=dict_class_names

# set dicts to the library
birds_lib.set_dicts(bird_dicts)

```

### Functions

The multimodal features library contains powerful functions to support multimodal features use. Here are some implemented functions. 

#### Aligning multimodal features

Due to the time properties of `computational_sequence` object, if two object's features' timeline is not the same, then the library provides the function to align them from one to another. 

Here is the example:

1. Let's define a function to create fake MMFLibs:

```python
from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
import numpy as np
from mmkfeatures.fusion.dataset import mmdataset

def create_fake_mmf_lib(dataset_name,feature_num,content_num,output_csd_file):
    # 1. create a empty multimodal features library (mmf_lib) with root_name and dataset_names(default saving file name)
    feature_lib = MMFeaturesLib(root_name=dataset_name, dataset_name=dataset_name)
    # 2. set names for each dimension name for convenience
    fake_features_name=["feature"+str(i+1) for i in range(feature_num)]
    feature_lib.set_features_name(fake_features_name)
    # 3. set a  list of content ids
    fake_content_ids=["content"+str(i+1) for i in range(content_num)]
    content_ids =fake_content_ids
    # 4. according to ids, assign a group of features with interval to corresponding content id
    features_dict = {}
    for id in content_ids:
        features = []
        intervals = []
        count = 0
        for i in range(5):
            feature = np.array([1, 2, 3])
            features.append(feature)
            interval = [count, count + 1]
            count += 1
            intervals.append(interval)
        features_dict[id] = [features, intervals]
    # 5. set data to the above feature dictionary with content id key
    feature_lib.set_data(features_dict)
    # 6. you can save the features to disk
    feature_lib.save_data(output_csd_file)
    # 7. you can check structure of csd file which happens to be the format of h5dy
    feature_lib.show_structure(output_csd_file)
    # 8. have a glance of features content within the dataset
    feature_lib.show_sample_data(output_csd_file)
    return feature_lib
```

2. Then, with the use of `create_fake_mmf_lib` functoin, we create two libraries and try to align them as follows:

```python
if __name__ == "__main__":

    lib1=create_fake_mmf_lib("test7_1",5,10,"test7_1.csd")
    lib2 = create_fake_mmf_lib("test7_2", 3, 8, "test7_2.csd")

    # 9. align to the dataset
    # now creating a toy dataset from the toy compseqs
    mydataset_recipe = {"mmf1": "test7_1.csd", "mmf2": "test7_2.csd"}
    mydataset = mmdataset(mydataset_recipe)
    # let's also see if we can align to compseq_1
    mydataset.align("mmf1")
```

#### Indexing and Searching

We also can use the library to index and search key fields from the multimodal data. The index content be multimodal content, like text-based and image-like features. 

1. Create a text index with different types of indexing techniques:

```python
from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
mmf_file=f"icd11_mmf/icd11_rel.mmf"
print("loading mmf files...")
feature_lib=MMFeaturesLib(file_path=mmf_file)
print("creating plain text index...")
feature_lib.to_index_file("text","icd11_mmf/index_test/text.index",index_type="brutal_force")

```
The `index_type` value can be brutal_force, inverted_index and positional_index. 

2. Then, we can use the index file to search something, and obtain the content ID. 

```python
print("searching plain index test....")
result_bf=feature_lib.search_index(index_file_path="icd11_mmf/index_test/text.index",query="Fungal infection",search_type="brutal_force")
print(result_bf)
```

3. We can also export image-based index and search image-index file
```python
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

```

The `index_type` in image_index search has values of `color_descriptor` or `autoencoder`. 

#### Showing structures and sample data of the library

The library also allows us to understand the structure of the database before use and get a small number of sample data from the database to help us understand better in its multimodal contents. 

1. First, load the multimodal feature library:

```python
from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib

mmf_file=f"datasets/birds.mmf"

feature_lib=MMFeaturesLib(file_path=mmf_file)
```

2. Then, call `show_structure()` and `show_sample_data()` from the library.

```python
feature_lib.show_structure()

feature_lib.show_sample_data(max_num=5)
```

3. Or you can iterate all data elements from the library. 

```
for key in data.keys():
    content=data[key]
    print(key)
```