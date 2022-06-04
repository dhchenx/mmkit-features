import objgraph
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

    objgraph.show_refs(feature_lib,filename='obj-graph.png')

