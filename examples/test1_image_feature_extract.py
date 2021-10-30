from mmkfeatures.image.image_features_wrapper import ImageFeaturesWrapper

def test_extract_deep_features():
    img_feat_wrapper = ImageFeaturesWrapper()
    # 1. extract a single feature from a file
    src = "../data/micro-imagenet/train/n03014705/images/n03014705_0.JPEG"
    dest = None
    features = img_feat_wrapper.extract_deep_features( src, dest)
    print(features[0].shape)
    print(features)

    # 2. extract features from a folder and save the features to a file
    src = "../data/micro-imagenet/train"
    dest = "../data/output/feature.csv"
    features = img_feat_wrapper.extract_deep_features(src, dest)

def test_extract_lbp_features():
    img_feat_wrapper = ImageFeaturesWrapper()
    # 1. extract a single feature from a file
    src = "../data/micro-imagenet/train/n03014705/images/n03014705_0.JPEG"
    dest = None
    features=img_feat_wrapper.extract_lbp_features(src,dest)
    print(features[0].shape)
    print(features)

    # 2. extract features from a folder and save the features to a file
    src = "../data/micro-imagenet/train"
    dest = "../data/output/feature.csv"
    features=img_feat_wrapper.extract_lbp_features(src,dest)

def test_extract_bow_features():
    img_feat_wrapper = ImageFeaturesWrapper()
    # 1. extract a single feature from a file
    src_tiny_imagenet = "../data/micro-imagenet/train" # the src file should be in the list of the src_tiny_imagenet file collection
    src = "../data/micro-imagenet/train/n03014705/images/n03014705_0.JPEG"
    dest = None
    features = img_feat_wrapper.extract_bow_features(src_tiny_imagenet, src, dest) # extrat the features of the src file upon the folder of *src_tiny_imagenet*
    print(features[0].shape)
    print(features)

    # 2. extract features from a folder and save the features to a file
    src = "../data/micro-imagenet/train"
    dest = "../data/output/feature.csv"
    features = img_feat_wrapper.extract_bow_features(src_tiny_imagenet, src, dest)

if __name__ == '__main__':
    # test_extract_deep_features()
    # test_extract_lbp_features()
    test_extract_bow_features()

