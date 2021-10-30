import random

from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
from mmkfeatures.image.image_autoencocer_builder import ImageAutoEncoderBuilder
from tqdm import tqdm
import numpy as np
import cv2

# load an existing multimodal feature lib
mmf_file=f"datasets/birds_raw.mmf"
feature_lib=MMFeaturesLib(file_path=mmf_file)
data=feature_lib.get_data()

# build the autoencoder for given images
img_autoencoder_builder=ImageAutoEncoderBuilder(img_size=(64,64),channel=3,epochs=20,split=0.8)

# obtain train and test datsets
trainX,testX,_,_=img_autoencoder_builder.create_train_test_ids(data,is_shuffle=True)

# train the model
img_autoencoder_builder.train(trainX,testX, output_model_path="autoencoder/output/autoencoder.h5")

# evaluate
# img_autoencoder_builder.predict(testX=testX, _vis="autoencoder/output/recon_vis.png",_plot="autoencoder/output/plot.png")
image_pairs=img_autoencoder_builder.predict_image(testX=testX, samples=5, _vis="autoencoder/output/recon_vis.png",_plot="autoencoder/output/plot.png")


# original image
original=image_pairs[0][0]

# reconstructed image
reconstructed=image_pairs[0][1]

cv2.imshow("image features",reconstructed)
cv2.waitKey(0)

# show
