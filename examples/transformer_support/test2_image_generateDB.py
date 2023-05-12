import os
import argparse

import torch
import cv2.cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from mmkfeatures.transformer.image.config import get_config
from mmkfeatures.transformer.image.build import build_model
from mmkfeatures.transformer.image.utils import get_pretrained_model
from tqdm import tqdm

def parse_option():
    parser = argparse.ArgumentParser(
        'establish database script', add_help=False)
    parser.add_argument('--cfg', type=str,
                        metavar="FILE", help='path to config file',default='datasets/image1/swin_tiny_patch4_window7_224.yaml' )
    parser.add_argument('--data-path', type=str, help='path to dataset', default=r'datasets/image1/cub_data/images')
    parser.add_argument('--batch-size', type=int, default=1,
                        help="batch size for single GPU")
    parser.add_argument(
        '--resume', help='resume from checkpoint', default="datasets/image1/swin_tiny_patch4_window7_224.pth")

    # distributed training
    parser.add_argument("--local_rank", type=int,help='local rank for DistributedDataParallel',default=1)

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def traverse_get_index_file(dataset_path, index_file_path):
    """
    Traverse `dataset_path` and save all image paths to file `index_file_path`.
    """
    if os.path.exists(index_file_path):
        return

    print(f"=> start traversing dataset path")
    with open(index_file_path, 'w', encoding='UTF-8') as f:
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.jpg'):  # change here to suit more formats image
                    f.write(f'{os.path.join(root, file)}\n')
        print(f"=> generate index file ({index_file_path}) sucessfully")


def establish_feat_database(model, dataloader_, db_path, device):
    """
    Extract features of all images by model, then save them to `db_path` with npz format.


    Example of reading `DB.npz` file:
    ```python
    import numpy as np

    kv = np.load('DB.npz', allow_pickle=True)
    feats, indexes = kv['DATA'], kv['INDEX']

    print(f'feats: {feats.shape}  indexes: {indexes.shape}')
    ```
    """
    db_feat, db_index = [], []
    with torch.no_grad():
        for batch_ndx, (samples, paths) in tqdm(enumerate(dataloader_)):
            # samples: torch.Tensor, shape (b, c, h, w); paths: list, shape (b, )
            # print(paths)
            try:
                samples = samples.to(device)

                output = model(samples)
                db_feat.append(output)

                db_index.extend(paths)
            except:
                print("ERROR")
                continue

    db_feat = torch.vstack(db_feat).cpu().numpy()
    db_index = np.array(db_index)

    np.savez(db_path, DATA=db_feat, INDEX=db_index)
    print(
        f"=> establish feature database ({db_path}) with shape ({db_feat.shape}) sucessfully")


class _DATASET(Dataset):
    def __init__(self, index_path='datasets/image1/index2.txt', IMG_SIZE=224, transform=None):
        super().__init__()
        self.IMG_SIZE = IMG_SIZE
        self.table = np.loadtxt(index_path, dtype=str)
        self.transform = transform

    def __getitem__(self, index):
        path_ = self.table[index]
        img = plt.imread(path_)
        img = cv.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        if self.transform:
            img = self.transform(img)
        return img, path_

    def __len__(self):
        return len(self.table)


def main():
    _, config = parse_option()

    # traverse path and generate images path index file with txt format
    dataset_path, index_file_path = config.DATA.DATABASE_PATH, config.DATA.INDEX_PATH
    dataset_path=r"datasets/image1/cub_data/images"
    index_file_path="datasets/image1/index2.txt"
    print(dataset_path,index_file_path)
    traverse_get_index_file(dataset_path, index_file_path)

    # build dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_ = _DATASET(index_file_path, transform=transform)
    dataloader_ = torch.utils.data.DataLoader(
        dataset_,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    # get model
    model = get_pretrained_model(config)

    # establish feature database
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    establish_feat_database(
        model, dataloader_, config.DATA.DATABASE_PATH, device)


if __name__ == '__main__':
    import time
    start=time.time()
    main()
    print("Create Time: ",time.time()-start)
