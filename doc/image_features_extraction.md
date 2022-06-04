## Image Features Extraction

The module aims to provide commonly used methods to obtain image features for convenience of use. The implmented methods in this toolki are far from comprehensive; therefore, users can implement their image feature extracting methods for specific purposes. 

The types of image feature include deep learning-based features, LBP features and BOW features using ImageNet. 

### 1. Deep learning-based features extraction 

Here is a toy example to extract deep learning-based features for images. 

A toy example is below: 

```python
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
```

### 2. LBP features extraction

A toy example is below:

```python
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
```

### 3. BOW features extraction

A toy example is here:

```python
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

```

The basic usage of the `mmkit-features` toolkit is to generate multimodal features and store the features for future use. Therefore, the extracted features should later be stored in a well-formatted way. The methods used above give a simple way to store the extracted features to the csv plain-text file, sometimes it is useful but may be not efficient. 