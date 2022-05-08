from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
from mmkfeatures.image.color_descriptor import ColorDescriptor
import numpy as  np
import cv2
import pickle
from tqdm import tqdm
import time

mmf_file=f"../datasets/birds_raw.mmf"

feature_lib=MMFeaturesLib(file_path=mmf_file)

data=feature_lib.get_data()

cd = ColorDescriptor((8, 12, 3))

list_features=[]

print(data.keys())
start_time=time.time()
f_out=open("../datasets/image.index","w")
for cid in tqdm(data.keys()):
    item=data[cid]
    imgs=item["objects"]
    for img_id in imgs:
        # print(image)
        img=imgs[img_id][()]
        # print(img)
        # print(type(img))
        features = cd.describe(img)
        features = [str(f) for f in features]
        # print(feature)
        feature_str=cid+","+",".join(features)
        f_out.write(feature_str+"\n")
f_out.close()
end_time=time.time()
time_cost=end_time-start_time
print("creating image index: ",time_cost)