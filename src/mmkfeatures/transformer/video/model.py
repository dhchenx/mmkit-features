# from tensorflow_docs.vis import embed
from keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os

"""
## Building the Transformer-based model

We will be building on top of the code shared in
[this book chapter](https://livebook.manning.com/book/deep-learning-with-python-second-edition/chapter-11) of
[Deep Learning with Python (Second ed.)](https://www.manning.com/books/deep-learning-with-python)
by François Chollet.

First, self-attention layers that form the basic blocks of a Transformer are
order-agnostic. Since videos are ordered sequences of frames, we need our
Transformer model to take into account order information.
We do this via **positional encoding**.
We simply embed the positions of the frames present inside videos with an
[`Embedding` layer](https://keras.io/api/layers/core_layers/embedding). We then
add these positional embeddings to the precomputed CNN feature maps.
"""

"""
Calling `prepare_all_videos()` on `train_df` and `test_df` takes ~20 minutes to
complete. For this reason, to save time, here we download already preprocessed NumPy arrays:
"""


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask


"""
Now, we can create a subclassed layer for the Transformer.
"""


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation=tf.nn.gelu),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


"""
## Utility functions for training
"""

class TransformerVideoFeatureExtractor:
    def __init__(self,root_folder,max_seq_length=20,num_features=1024,num_epochs=100,img_size=128):
        self.MAX_SEQ_LENGTH = max_seq_length
        self.NUM_FEATURES = num_features
        self.IMG_SIZE = img_size
        self.EPOCHS = num_epochs
        self.root_folder=root_folder

    def load_data(self):
        self.train_df = pd.read_csv(f"{self.root_folder}/train.csv")
        self.test_df = pd.read_csv(f"{self.root_folder}/test.csv")

        self.label_processor = keras.layers.StringLookup(
            num_oov_indices=0, vocabulary=np.unique(self.train_df["tag"]), mask_token=None
        )
        print(self.label_processor.get_vocabulary())

        self.class_vocab = self.label_processor.get_vocabulary()

        self.train_data, self.train_labels = np.load(f"{self.root_folder}/train_data.npy"), np.load(f"{self.root_folder}/train_labels.npy")
        self.test_data, self.test_labels = np.load(f"{self.root_folder}/test_data.npy"), np.load(f"{self.root_folder}/test_labels.npy")

        print(f"Frame features in train set: {self.train_data.shape}")

        self.center_crop_layer = layers.CenterCrop(self.IMG_SIZE, self.IMG_SIZE)

    def get_compiled_model(self):
        sequence_length = self.MAX_SEQ_LENGTH
        embed_dim = self.NUM_FEATURES
        dense_dim = 4
        num_heads = 1
        classes = len(self.label_processor.get_vocabulary())

        inputs = keras.Input(shape=(None, None))
        x = PositionalEmbedding(
            sequence_length, embed_dim, name="frame_position_embedding"
        )(inputs)
        x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        return model

    def run_experiment(self):
        filepath = "video_classifier"
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath, save_weights_only=True, save_best_only=True, verbose=1
        )

        model = self.get_compiled_model()
        history = model.fit(
            self.train_data,
            self.train_labels,
            validation_split=0.15,
            epochs=self.EPOCHS,
            callbacks=[checkpoint],
        )

        model.load_weights(filepath)
        _, accuracy = model.evaluate(self.test_data, self.test_labels)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")

        return model, accuracy

    def crop_center(self,frame):
        cropped = self.center_crop_layer(frame[None, ...])
        cropped = cropped.numpy().squeeze()
        return cropped

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

    def prepare_single_video(self,frames):
        frame_features = np.zeros(shape=(1, self.MAX_SEQ_LENGTH, self.NUM_FEATURES), dtype="float32")

        # Pad shorter videos.
        if len(frames) < self.MAX_SEQ_LENGTH:
            diff = self.MAX_SEQ_LENGTH - len(frames)
            padding = np.zeros((diff, self.IMG_SIZE, self.IMG_SIZE, 3))
            frames = np.concatenate(frames, padding)

        frames = frames[None, ...]

        self.feature_extractor=self.build_feature_extractor()

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(self.MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                if np.mean(batch[j, :]) > 0.0:
                    frame_features[i, j, :] = self.feature_extractor.predict(batch[None, j, :])
                else:
                    frame_features[i, j, :] = 0.0

        return frame_features


    def predict_action(self,video_path):
        frames = self.load_video(video_path)
        frame_features = self.prepare_single_video(frames)
        probabilities = self.trained_model.predict(frame_features)[0]
        dict_result={}
        for i in np.argsort(probabilities)[::-1]:
            # print(f"  {self.class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
            dict_result[self.class_vocab[i]] = probabilities[i]
        return dict_result

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

    # This utility is for visualization.
    # Referenced from:
    # https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
    def to_gif(self,images):
        converted_images = images.astype(np.uint8)
        imageio.mimsave("animation.gif", converted_images, fps=10)
        # return embed.embed_file("animation.gif")

    def train(self):
        """
                ## Model training and inference
                """
        self.trained_model, self.accuracy = self.run_experiment()

    def predict(self,test_video):
        """
        **Note**: This model has ~4.23 Million parameters, which is way more than the sequence
        model (99918 parameters) we used in the prequel of this example.  This kind of
        Transformer model works best with a larger dataset and a longer pre-training schedule.
        """
        # test_video = np.random.choice(self.test_df["video_name"].values.tolist())
        print(f"Test video path: {test_video}")
        results = self.predict_action(test_video)

        # self.to_gif(test_frames[:self.MAX_SEQ_LENGTH])
        return results

