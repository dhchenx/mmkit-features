

import os

# os.system("python main.py feature_type=pwc device_ids=[0] video_paths=[../../../../data/videos/a.mp4]")
import time

from mmkfeatures.video.models.main import *

sizes=["240m","240","360","480","720","1280"]
list_result=[]
for size in sizes:
    feature_types=['i3d','pwc','r21d','raft','resnet18','vggish']

    for f in feature_types:
        start=time.time()
        extract_video_features_by_model(feature_type=f,video_path=f'videos/giraffes_{size}.mp4')
        end=time.time()
        cost=end-start
        list_result.append([size,f,cost])


print()

print("size\tfeature type\ttime cost")
for idx,f in enumerate(list_result):
    print(f"{f[0]}\t{f[1]}\t{f[2]}")