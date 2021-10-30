from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
from mmkfeatures.image.color_descriptor import ColorDescriptor
import numpy as  np
import cv2
import pickle
from tqdm import tqdm
from mmkfeatures.image.image_searcher import Searcher

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

