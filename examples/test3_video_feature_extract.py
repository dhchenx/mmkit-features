
from mmkfeatures.video.video_features_wrapper import  VideoFeatureWrapper

if __name__ == "__main__":
    video_feat_wrapper=VideoFeatureWrapper()
    video_file = "../data/videos/a.mp4"
    output_file = "../data/output/single_features.npy"
    features=video_feat_wrapper.extract_video_features(video_file,output_file)
    print(features)
    print(features.shape)

