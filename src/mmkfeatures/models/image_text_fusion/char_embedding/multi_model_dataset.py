import torch
from torch.utils.data import Dataset
from mmkfeatures.models.image_text_fusion.char_embedding.char_cnn_rnn import prepare_text


class MultimodalDataset(Dataset):
    # Arguments:
    # dataset_json (json): json contains dataset item paths.
    # data_folder (String): dataset root path.
    def __init__(self, dataset_json, data_folder, device):
        super().__init__()
        self.dataset_json = dataset_json
        self.data = []
        self.device = device
        for item in self.dataset_json:
            # Load saved image tensor and convert
            # text to encoded tensor, size 1024
            image = torch.load(data_folder + item['img'])
            image.requires_grad = False
            text = prepare_text(item['text'])
            image = image.to(self.device)
            text = text.to(self.device)
            self.data.append({'img': image, 'text': text})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
