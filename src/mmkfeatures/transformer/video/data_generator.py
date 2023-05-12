from keras import layers
from tensorflow import keras
import pandas as pd
import numpy as np
import imageio
import cv2
import os

"""
## Data preparation

We will mostly be following the same data preparation steps in this example, except for
the following changes:

* We reduce the image size to 128x128 instead of 224x224 to speed up computation.
* Instead of using a pre-trained [InceptionV3](https://arxiv.org/abs/1512.00567) network,
we use a pre-trained
[DenseNet121](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)
for feature extraction.
* We directly pad shorter videos to length `MAX_SEQ_LENGTH`.

First, let's load up the
[DataFrames](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html).
"""

"""
## Define hyperparameters
"""

class VideoDatasets:

    def __init__(self,root_folder="ucf101_top5"):
        self.MAX_SEQ_LENGTH = 20
        self.NUM_FEATURES = 1024
        self.IMG_SIZE = 128
        self.EPOCHS = 100
        self.root_folder=root_folder

    def load_data(self):

        self.train_df = pd.read_csv(f"{self.root_folder}/train.csv")
        self.test_df = pd.read_csv(f"{self.root_folder}/test.csv")

        print(f"Total videos for training: {len(self.train_df)}")
        print(f"Total videos for testing: {len(self.test_df)}")

        self.center_crop_layer = layers.CenterCrop(self.IMG_SIZE, self.IMG_SIZE)

    def crop_center(self,frame):
        cropped = self.center_crop_layer(frame[None, ...])
        cropped = cropped.numpy().squeeze()
        return cropped

    # Following method is modified from this tutorial:
    # https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
    def load_video(self,path, max_frames=0):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.crop_center(frame)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

                if len(frames) == max_frames:
                    break
        finally:
            cap.release()
        return np.array(frames)


    def build_feature_extractor(self):
        feature_extractor = keras.applications.DenseNet121(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
        )
        preprocess_input = keras.applications.densenet.preprocess_input

        inputs = keras.Input((self.IMG_SIZE, self.IMG_SIZE, 3))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        return keras.Model(inputs, outputs, name="feature_extractor")

    def generate(self):
        self.feature_extractor = self.build_feature_extractor()
        # Label preprocessing with StringLookup.
        self.label_processor = keras.layers.StringLookup(
            num_oov_indices=0, vocabulary=np.unique(self.train_df["tag"]), mask_token=None
        )
        print(self.label_processor.get_vocabulary())

    def prepare_all_videos(self,label_processor, df, root_dir):
        num_samples = len(df)
        video_paths = df["video_name"].values.tolist()
        labels = df["tag"].values
        labels = label_processor(labels[..., None]).numpy()

        # `frame_features` are what we will feed to our sequence model.
        frame_features = np.zeros(
            shape=(num_samples, self.MAX_SEQ_LENGTH, self.NUM_FEATURES), dtype="float32"
        )

        # For each video.
        for idx, path in enumerate(video_paths):
            # Gather all its frames and add a batch dimension.
            frames = self.load_video(os.path.join(root_dir, path))

            # Pad shorter videos.
            if len(frames) < self.MAX_SEQ_LENGTH:
                diff = self.MAX_SEQ_LENGTH - len(frames)
                padding = np.zeros((diff, self.IMG_SIZE, self.IMG_SIZE, 3))
                frames = np.concatenate(frames, padding)

            frames = frames[None, ...]

            # Initialize placeholder to store the features of the current video.
            temp_frame_features = np.zeros(
                shape=(1, self.MAX_SEQ_LENGTH, self.NUM_FEATURES), dtype="float32"
            )

            # Extract features from the frames of the current video.
            for i, batch in enumerate(frames):
                video_length = batch.shape[0]
                length = min(self.MAX_SEQ_LENGTH, video_length)
                for j in range(length):
                    if np.mean(batch[j, :]) > 0.0:
                        temp_frame_features[i, j, :] = self.feature_extractor.predict(
                            batch[None, j, :]
                        )

                    else:
                        temp_frame_features[i, j, :] = 0.0

            frame_features[
                idx,
            ] = temp_frame_features.squeeze()

        return frame_features, labels

    def save_data(self):

        train_data, train_labels= self.prepare_all_videos(df=self.train_df,root_dir=f'{self.root_folder}/train',label_processor=self.label_processor)
        test_data, test_labels = self.prepare_all_videos(df=self.test_df,root_dir=f'{self.root_folder}/test',label_processor=self.label_processor)

        np.save(f"{self.root_folder}/train_data.npy", train_data, fix_imports=True, allow_pickle=False)
        np.save(f"{self.root_folder}/train_labels.npy", train_labels, fix_imports=True, allow_pickle=False)
        np.save(f"{self.root_folder}/test_data.npy", test_data, fix_imports=True, allow_pickle=False)
        np.save(f"{self.root_folder}/test_labels.npy", test_labels, fix_imports=True, allow_pickle=False)

if __name__=="__main__":
    vd=VideoDatasets(root_folder='../ucf101_top5')
    vd.load_data()
    vd.generate()
    vd.save_data()