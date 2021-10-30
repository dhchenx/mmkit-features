from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
from mmkfeatures.image.color_descriptor import ColorDescriptor
import numpy as  np
import cv2
import pickle
from tqdm import tqdm
from mmkfeatures.image.image_searcher import Searcher

mmf_file=f"datasets/birds_raw.mmf"

feature_lib=MMFeaturesLib(file_path=mmf_file)

data=feature_lib.get_data()


# initialize the image descriptor
cd = ColorDescriptor((8, 12, 3))

img_path="datasets/CUB_200_2011/images/005.Crested_Auklet/Crested_Auklet_0001_794941.jpg"
index_path="datasets/image.index"

print("Searching....")
# load the query image and describe it
query = cv2.imread(img_path)
features = cd.describe(query)
# perform the search
searcher = Searcher(index_path)
results = searcher.search(features)
# display the query
# cv2.imshow("Query", query)
# cv2.waitKey(0)

# loop over the results
for (score, resultID) in results:
    print(resultID,score)
    image=data[str(resultID)]["objects"]["0"][()]
    title=data[str(resultID)]["labels"][()][0]
    cv2.imshow(title, image)
    cv2.waitKey(0)
    # load the result image and display it
    # result = cv2.imread("datasets/CUB_200_2011/" + resultID)
    # cv2.imshow("Result", result)
    # cv2.waitKey(0)
