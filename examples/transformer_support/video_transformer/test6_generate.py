from mmkfeatures.transformer.video.data_generator import *

if __name__=="__main__":
    '''
    Create video datasets suitable for training and testing
    '''
    vd=VideoDatasets(root_folder='ucf101_top5')
    vd.load_data()
    vd.generate()
    vd.save_data()