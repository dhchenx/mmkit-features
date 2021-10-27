import glob
import cv2

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # encoder layers ##
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adapt_pool = nn.AdaptiveMaxPool2d((16, 16))

        # decoder layers ##
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x, encoder_mode=False):
        # encode ##
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x_encoder = self.adapt_pool(x)

        if encoder_mode:
            return x_encoder
        else:
            # decode ##
            x = F.relu(self.t_conv1(x))
            x = F.sigmoid(self.t_conv2(x))
            return x, x_encoder


class ImgDataset(Dataset):
    def __init__(self, data_root, transform, device, img_format='.jpg'):
        super(ImgDataset, self).__init__()
        self.sample = []
        self.transform = transform
        self.device = device
        self.img_format = img_format
        print(f"ImgDataset load images from - {data_root + '/*' + img_format}")
        for img in glob.glob(data_root + '/*' + img_format):
            image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            image = self.transform(image)
            image = image.float()
            image = image.to(self.device)
            self.sample.append(image)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, index):
        return self.sample[index]
