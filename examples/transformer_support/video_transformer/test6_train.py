from mmkfeatures.transformer.video.model import *

if __name__=="__main__":
    '''
    Training and testing Transform-based Video Feature Extractor
    '''
    vfe = TransformerVideoFeatureExtractor(root_folder='ucf101_top5',num_epochs=10)
    vfe.load_data()
    vfe.train()
    results=vfe.predict(test_video='ucf101_top5/test/v_CricketShot_g01_c01.avi')
    for k in results:
        print(f"{k}:{round(results[k],4)}")