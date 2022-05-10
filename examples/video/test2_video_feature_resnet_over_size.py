import time

from mmkfeatures.video.models.main import *

sizes=["240m","240","360","480","720","1280"]
list_result=[]
for size in sizes:
    resnet_types=['resnet18','resnet34','resnet50','resnet101','resnet152']
    for r in resnet_types:
        start = time.time()
        extract_video_features_by_model(feature_type=r, video_path=f'videos/giraffes_{size}.mp4')
        end = time.time()
        cost = end - start
        list_result.append([r,size,cost])

print()
print("resnet type\tsize\ttime cost")
for idx,f in enumerate(list_result):
    print(f"{f[0]}\t{f[1]}\t{f[2]}")