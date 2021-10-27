import os
from mmkfeatures.image.deep_extractor import DeepExtractor
from mmkfeatures.image.lbp_extractor import LBPExtractor
from mmkfeatures.image.bow_extractor import BoWExtractor

# model_name options: ['vgg19', 'inception_v3', 'inception_resnet_v2']

class ImageFeaturesWrapper:
    def __init__(self):
        pass

    def get_deep_feature_extractor(self,src,dest=None,model_name="vgg19"):
        return DeepExtractor(base_route=src, model_name=model_name, size=75)

    def extract_deep_features(self,src, extractor=None,dest=None,model_name="vgg19"):
        if extractor==None:
            extractor = DeepExtractor(base_route=src, model_name=model_name, size=75)

        features=[]
        if dest==None:
            if os.path.isfile(src):
                features = [extractor.extract(src)]
            elif os.path.isdir(src):
                features = []
                list = os.listdir(src)  # 列出文件夹下所有的目录与文件
                for i in range(0, len(list)):
                    path = os.path.join(src, list[i])
                    if os.path.isfile(path):
                        features.append(extractor.extract(path))
            else:
                raise Exception("No file path is recognized!")
            return features
        else:
            if not os.path.isdir(src):
                raise Exception("src should be a directory when dest has value. ")

            extractor.extract_and_save(dest)
            return features


    def extract_lbp_features(self,src,dest=None,size=64,points=3,radius=1,x=2,y=4):

        extractor = LBPExtractor(base_route=src, size=size, points=points, radius=radius, grid_x=x, grid_y=y)

        features = []
        if dest==None:
            if os.path.isfile(src):
                features = [extractor.extract(src)]
            elif os.path.isdir(src):
                features = []
                list = os.listdir(src)  # 列出文件夹下所有的目录与文件
                for i in range(0, len(list)):
                    path = os.path.join(src, list[i])
                    if os.path.isfile(path):
                        features.append(extractor.extract(path))
            else:
                raise Exception("No file path is recognized!")
            return features
        else:
            if not os.path.isdir(src):
                raise Exception("src should be a directory when dest has value. ")

            extractor.extract_and_save(dest)
            return features

    def extract_bow_features(self,src_dir,src,dest,k=4, method="kaze", size=64, cluster_mode='manual'):

        extractor = BoWExtractor(base_route=src_dir, k=k, method=method, size=size,
                                 cluster_mode=cluster_mode)
        extractor.setup()
        extractor.fit()

        features = []
        if dest==None:
            if os.path.isfile(src):
                features = [extractor.extract(src)]
            elif os.path.isdir(src):
                features = []
                list = os.listdir(src)  # 列出文件夹下所有的目录与文件
                for i in range(0, len(list)):
                    path = os.path.join(src, list[i])
                    if os.path.isfile(path):
                        features.append(extractor.extract(path))
            else:
                raise Exception("No file path is recognized!")
            return features
        else:
            if not os.path.isdir(src):
                raise Exception("src should be a directory when dest has value. ")
            extractor.extract_and_save(dest)
            return features

if __name__ == '__main__':
    img_feat_wrapper=ImageFeaturesWrapper()
    # 1. extract a single feature from a file
    src_tiny_imagenet="../dataset/micro-imagenet/train"
    src="../dataset/micro-imagenet/train/n03014705/images/n03014705_0.JPEG"
    dest=None
    # features=img_feat_wrapper.extract_lbp_features(src,dest)
    # features=img_feat_wrapper.extract_bow_features(src,dest)
    features=img_feat_wrapper.extract_bow_features(src_tiny_imagenet,src,dest)
    print(features[0].shape)
    print(features)

    # 2. extract features from a folder and save the features to a file
    src="../dataset/micro-imagenet/train"
    dest="../dataset/output/feature.csv"
    # features=img_feat_wrapper.extract_lbp_features(src,dest)
    features = img_feat_wrapper.extract_bow_features(src_tiny_imagenet,src, dest)

