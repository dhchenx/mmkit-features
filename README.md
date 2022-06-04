# MMKit-Features: Multimodal Features Extraction Toolkit

A light-weight Python library to extract, fuse and store multimodal features for deep learning.

## Objectives
- To extract, store and fuse various features from multimodal datasets rapidly and efficiently;
- To provide a common multimodal information processing framework for multimodal features; 
- To achieve generative adversarial network (GAN)-based multimodal knowledge representation dynamically. 

## Modalities
1. Text/Language modality
2. Image modality
3. Video modality
4. Speech/sound modality
5. Cross-modality between above

## Usage
A toy example showing how to build a multimodal feature (MMF) library is here:

```python
from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
from mmkfeatures.fusion.mm_features_node import MMFeaturesNode
import numpy as np
if __name__ == "__main__":
    # 1. create an empty multimodal features library with root and dataset names
    feature_lib=MMFeaturesLib(root_name="test features",dataset_name="test_features")
    # 2. set short names for each dimension for convenience
    feature_lib.set_features_name(["feature1","feature2","feature3"])
    # 3. set a  list of content IDs
    content_ids=["content1","content2","content3"]
    # 4. according to IDs, assign a group of features with interval to corresponding content ID
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

Further instructions on the toolkit refers to [here](https://github.com/dhchenx/mmkit-features/tree/main/doc). 

## Credits
The project includes some source codes from various open-source contributors. Here is a list of their contributions. 

1. [A2Zadeh/CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK)
2. [aishoot/Speech_Feature_Extraction](https://github.com/aishoot/Speech_Feature_Extraction)
3. [antoine77340/video_feature_extractor](https://github.com/antoine77340/video_feature_extractor)
4. [jgoodman8/py-image-features-extractor](https://github.com/jgoodman8/py-image-features-extractor)
5. [v-iashin/Video Features](https://v-iashin.github.io/video_features/)

## License
Please consider cite our project if the project is used in your research. 

## To do

2. Multimodal knowledge base framework
3. Graphic User Interface for the framework
...
