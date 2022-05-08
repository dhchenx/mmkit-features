import random
import time
from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
from mmkfeatures.image.image_autoencocer_builder import ImageAutoEncoderBuilder

# load an existing multimodal feature lib
mmf_file=f"datasets/birds_raw.mmf"
feature_lib=MMFeaturesLib(file_path=mmf_file)
data=feature_lib.get_data()

# build the autoencoder for given images

img_autoencoder_builder=ImageAutoEncoderBuilder(img_size=(64,64),channel=3,epochs=20,split=0.8)

# load train and test datasets
trainX,testX,list_train,list_test = img_autoencoder_builder.create_train_test_ids(data,is_shuffle=True)

# load model
img_autoencoder_builder.load_model("autoencoder/output/autoencoder.h5")

# create index
all_ids,all_data=img_autoencoder_builder.create_datasets(data)
img_autoencoder_builder.create_index(all_ids,all_data,index_file="autoencoder/output/image_autoencoder.index")

# test search
# 1. search and display some results
# img_autoencoder_builder.search(index_file="autoencoder/output/image_autoencoder.index",queryXs=testX,display=True,Xs=all_data,display_num=5)
# 2. search image and obtain only ids.
query_subset=testX[:10]
start_time=time.time()
queryIds=img_autoencoder_builder.search(index_file="autoencoder/output/image_autoencoder.index",queryXs=query_subset,display=False)
end_time=time.time()
print("time cost with autoencoder: ",end_time-start_time)
for idx,item in enumerate(queryIds):
    print(f"Query {idx+1}:", item)