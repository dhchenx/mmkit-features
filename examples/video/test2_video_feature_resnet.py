import time

from mmkfeatures.video.models.main import *

resnet_types=['resnet18','resnet34','resnet50','resnet101','resnet152']
list_result=[]
for r in resnet_types:
    start = time.time()
    extract_video_features_by_model(feature_type=r, video_path='../../data/videos/a.mp4')
    end = time.time()
    cost = end - start
    list_result.append(cost)

print("feature type\ttime cost")
for idx,f in enumerate(resnet_types):
    print(f"{f}\t{list_result[idx]}")