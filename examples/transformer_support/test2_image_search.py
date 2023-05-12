import math
import argparse
import time
import cv2.cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt
from mmkfeatures.transformer.image.config import get_config
from mmkfeatures.transformer.image.utils import ROISelector, cos_similarity, get_pretrained_model, set_subplot_border
from mmkfeatures.transformer.image.build import build_model

class DB(object):
    def __init__(self, db_path='database/DB.npz'):
        self.path = db_path
        self._read_db(db_path)

    def _read_db(self, path):
        db = np.load(path)

        self.table = db['INDEX']
        self.feat = db['DATA']

    def __len__(self):
        return len(self.table)

    def database(self):
        return self.feat, self.table

class TransformerImageFeatureExtractor:
    def __init__(self,db_data_path='',data_image_path='database/cub_data/images',config_file='configs/swin_tiny_patch4_window7_224.yaml',checkpoints_file='checkpoints/swin_tiny_patch4_window7_224.pth'):
        self.option=self.parse_option(config_file,checkpoints_file,data_image_path)
        self.db_data_path=db_data_path

    def parse_option(self,config_file,checkpoints_file,db_image_path):
        parser = argparse.ArgumentParser(
            'image retrieve script', add_help=False)
        parser.add_argument('--cfg', default=config_file, type=str,
                            help='path to config file', )

        parser.add_argument('--resume', default=checkpoints_file,
                            help='resume from checkpoint')

        # todo: useless, to be deleted
        parser.add_argument("--local_rank", type=int,
                            help='local rank for DistributedDataParallel')
        parser.add_argument('--batch-size', type=int,
                            help="batch size for single GPU", default=1)
        parser.add_argument('--data-path', default=db_image_path,
                            type=str, help='path to dataset')

        args, unparsed = parser.parse_known_args()

        config = get_config(args)

        return args, config


    def extract_feat(self,config, model, img):
        """
        Extract feature of input image by `model`.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device="cpu"
        print(device)
        img = cv.resize(img, (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE))
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis]
        img = torch.from_numpy(img)
        img = img.to(device)

        with torch.no_grad():
            output = model(img)
        feat = output.cpu().numpy()
        return feat


    def retrieve(self,config, feat, n_return=5):
        """
        Retrive similar image in database based on image feature.

        Parameters:
            feat (np.array): feature of retrive target
            n_return (int): number of most similar images returned

        Returns:
            similaritys_sorted (list)
            imgs (list of np.ndarray): retrived results
        """
        db = DB(self.db_data_path)
        db_feat, db_path = db.database()
        n = len(db)

        similaritys = []
        for i in range(n):
            similaritys.append(cos_similarity(db_feat[i], feat))
        similaritys = np.array(similaritys)
        index = np.argsort(similaritys)[::-1]
        index = index[:n_return]
        similaritys_sorted = similaritys[index]

        # return top `n_return` similar images
        db_path_sorted = db_path[index]
        imgs = []
        for path_ in db_path_sorted:
            imgs.append(plt.imread(path_))
        return similaritys_sorted, imgs


    def show_images(self,ori_img, cropped_img, retrived_imgs, similarity, col=None):
        """
        Plot original image, cropped image, and retrived images in `col` columns.
        """
        assert len(retrived_imgs) == len(similarity), \
            f'Size of images {len(retrived_imgs)} should be same with size of sims {len(similarity)}.'
        n = len(retrived_imgs)
        col = int(col) if col else 5
        row = math.ceil(n / col) + 1
        h, w, _ = ori_img.shape
        h = w = min(h, w)

        # plot to show
        plt.figure()
        plt.subplots_adjust(wspace=.2, hspace=.2)
        for i in range(row):
            for j in range(col):
                idx = i * col + j
                ax = plt.subplot(row, col, idx + 1)

                if idx == 0:
                    title_, img_ = 'raw img', ori_img
                    set_subplot_border(ax, 'green', 4)
                elif idx == 1:
                    title_, img_ = 'cropped img', cropped_img
                    set_subplot_border(ax, 'red', 4)
                elif idx < col:
                    plt.axis('off')
                    continue
                else:
                    title_, img_ = f'{similarity[idx - col]:.4f}', retrived_imgs[idx - col]
                    set_subplot_border(ax, 'blue', 4)

                plt.title(title_)
                plt.xticks([])
                plt.yticks([])
                img_ = cv.resize(img_, (h, w))
                plt.imshow(img_)
        plt.show()


    def search(self,roi,show_fig=True,ori_img=None):
        _, config = self.option
        # 2. extract feature of roi
        start = time.time()
        model = get_pretrained_model(config)
        feat = self.extract_feat(config, model, roi)
        # print("Create Time:", time.time() - start)
        # 3. retrive in database
        # start=time.time()
        similarity, retrived_imgs = self.retrieve(config, feat, n_return=10)
        print("Search time:", time.time() - start)
        # 4. display
        if show_fig:
            self.show_images(ori_img, roi, retrived_imgs, similarity, col=5)


if __name__ == '__main__':
    root_path="datasets/image1"
    # 1. open image and select interested region
    # path = filedialog.askopenfilename()
    search_image_path = fr'{root_path}/sample2.jpg'
    roisor = ROISelector(search_image_path)
    plt.show()
    ori_img = roisor.img
    roi = roisor.cropped_img
    print(roi)

    ife=TransformerImageFeatureExtractor(db_data_path=f'{root_path}/DB.npz',
                                     data_image_path=f'{root_path}/cub_data/images',
                                     config_file=f'{root_path}/swin_tiny_patch4_window7_224.yaml',
                                     checkpoints_file=f'{root_path}/swin_tiny_patch4_window7_224.pth')
    # 2. Start to search
    ife.search(roi,show_fig=True,ori_img=ori_img)

