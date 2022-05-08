import os
import sys
from tqdm import tqdm
from mmkfeatures.image.image_processing import ImageUtils
from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
from mmkfeatures.fusion.mm_features_node import MMFeaturesNode
from mmkfeatures.image.image_features_wrapper import ImageFeaturesWrapper
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('INFO')
import time
start_time=time.time()
img_utils=ImageUtils()
root_path=r"D:\UIBE科研\国自科青年\多模态机器学习\projects\MedMNIST-main\examples\medmnist_data"

'''
img_feat_wrapper = ImageFeaturesWrapper()
deep_extractor=img_feat_wrapper.get_deep_feature_extractor("")
def get_deep_features(image_path):
    feat = img_feat_wrapper.extract_deep_features(image_path, extractor=deep_extractor, dest=None)
    return feat
'''

# start configure
flags=[f for f in os.listdir(root_path)]
# print(flags)


flag="pathmnist"
# end configure

# accept argument outside
if len(sys.argv)==2 and sys.argv[1] in flags:

    flag=sys.argv[1]
    print("accepting the flag:", flag)

list_data = {}
list_split = {}
splits="train,test"

for split in splits.split(','):
    # split = "test"

    data_path = root_path + "/" + flag
    specific_data_path = data_path + "/" + split + "/" + flag

    # get image data
    print(f"Processing {split} folder...")
    file_list=os.listdir(specific_data_path)
    for filename in tqdm(file_list):
        full_path = os.path.join(specific_data_path, filename)
        ext = os.path.splitext(full_path)[1]
        image_data = img_utils.get_numpy_array_rgb(full_path)
        # image_data=get_deep_features(full_path)
        list_data[filename] = image_data
        list_split[filename] = split
        # print(filename,ext)

# save to feature lib

# 1. create an empty multimodal features library (mmf_lib) with root_name and dataset_names(default saving file name)
feature_lib=MMFeaturesLib(root_name=f"{flag}",dataset_name=f"{flag}_features")
# 2. set names for each dimension name for convenience
# feature_lib.set_features_name(["feature1","feature2","feature3"])
# 3. set a  list of content ids
content_ids=list_data.keys()

# 4. according to ids, assign a group of features with interval to corresponding content id
features_dict={}
for id in content_ids:
    '''
    Multimodal Features Node Structure 
    
    Content Id:
        name: a short name for the content
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

    meta_fields={
        "name":"",
        "text":"",
        "modality":"image",
        "feature_ids":[],
        "feature_dim_names":[],
        "feature_extractor":"mmkit-features",
        "features":[],
        "intervals":[],
        "space":[],
        "labels":[],
        "origin":"",
        "attributes":{},
        "links":[]
    }
    # define property values
    features = []
    intervals = []
    features.append(list_data[id])
    intervals.append(np.array([]))
    filename = os.path.splitext(id)
    # setup a multimodal features node
    mmnode=MMFeaturesNode(content_id=filename[0])

    labels=[]
    labels.append(filename[1])
    mmnode.set_item("content_id","")
    mmnode.set_item("name",filename[0])
    # mmnode.set_item("format",filename[1])
    mmnode.set_item("modality","image")
    # mmnode.set_item("feature_ids",['f'+str(k+1) for k in range(len(features))])
    # mmnode.set_item("feature_extractor","numpy")
    mmnode.set_item("labels",labels)
    # mmnode.set_item("origin",id)
    mmnode.set_item("features",np.array(features))
    mmnode.set_item("intervals",np.array(intervals))
    # mmnode.set_item("attributes",{"split":list_split[id]})
    # add the node to a dictionary
    mmnode.validate_empty_field()
    features_dict[filename[0].split('_')[0]]=mmnode

# 5. set data to the above feature dictionary with content id key
feature_lib.set_data(features_dict)
# 6. you can save the features to disk
feature_lib.save_data(f"medmnist_mmf/{flag}.mmf")
# 7. you can check structure of csd file which happens to be the format of h5dy
# feature_lib.show_structure(f"medmnist_mmf/{flag}.mmf")
# 8. have a glance of features content within the dataset
# feature_lib.show_sample_data(f"medmnist_mmf/{flag}.mmf")

end_time=time.time()
time_cost=end_time-start_time
print("time cost: ",time_cost)