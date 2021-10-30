import numpy

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

if __name__ == "__main__":

    lib1=create_fake_mmf_lib("test7_1",5,10,"test7_1.csd")
    lib2 = create_fake_mmf_lib("test7_2", 3, 8, "test7_2.csd")

    # 9. align to the dataset
    # now creating a toy dataset from the toy compseqs
    mydataset_recipe = {"mmf1": "test7_1.csd", "mmf2": "test7_2.csd"}
    mydataset = mmdataset(mydataset_recipe)
    # let's also see if we can align to compseq_1
    mydataset.align("mmf1")


