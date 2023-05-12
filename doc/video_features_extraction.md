## Video Features Extraction

Extracting video features from a video file like `*.mp4` file is very complicated. There are many frames from the video which are considered as images. But at the same time, we have to consider the temporal information in the video. 

A simple example of extracting video features using the `mmkit-features` toolkit is below: 

```python

from mmkfeatures.video.video_features_wrapper import  VideoFeatureWrapper

if __name__ == "__main__":
    video_feat_wrapper=VideoFeatureWrapper()
    video_file = "../data/videos/a.mp4"
    output_file = "../data/output/single_features.npy"
    features=video_feat_wrapper.extract_video_features(video_file,output_file)
    print(features)
    print(features.shape)

```

The generated features are stored as numpy object to a npy file. The method will divided the video into n chunks and aggregated features from each chunk of video clips, so the final generated features are actually represented as a list of features of each chunk. 