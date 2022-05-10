import os

# os.system("python main.py feature_type=pwc device_ids=[0] video_paths=[../../../../data/videos/a.mp4]")
import time

from mmkfeatures.video.models.main import *

feature_types=['i3d','pwc','r21d','raft','resnet18','vggish']
list_result=[]
for f in feature_types:
    start=time.time()
    extract_video_features_by_model(feature_type=f,video_path='../../data/videos/a.mp4')
    end=time.time()
    cost=end-start
    list_result.append(cost)

print()

print("feature type\ttime cost")
for idx,f in enumerate(feature_types):
    print(f"{f}\t{list_result[idx]}")