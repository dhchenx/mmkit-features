import torch
import torch.nn as nn
from mmkfeatures.models.image_text_fusion.char_embedding.fixed_gru import FixedGru
from mmkfeatures.models.image_text_fusion.char_embedding.fixed_rnn import FixedRnn


class CharCnnRnn(nn.Module):
    # Char-cnn-rnn text embedding
    def __init__(self, rnn_type='fixed_rnn', model_type='cvpr'):
        super().__init__()
        if model_type == 'cvpr':
            rnn_dim = 256
            use_maxpool3 = True
            rnn = FixedRnn
            rnn_num_steps = 8
        else:
            # icml model type
            rnn_dim = 512
            if rnn_type == 'fixed_rnn':
                use_maxpool3 = True
                rnn = FixedRnn
                rnn_num_steps = 8
            else:
                # Use fixed_gru
                use_maxpool3 = False
                rnn = FixedGru
                rnn_num_steps = 18

        self.rnn_type = rnn_type
        self.model_type = model_type
        self.use_maxpool3 = use_maxpool3

        # network setup
        # (B, 70, 201)
        self.conv1 = nn.Conv1d(70, 384, kernel_size=4)
        self.threshold1 = nn.Threshold(1e-6, 0)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        # (B, 384, 66)
        self.conv2 = nn.Conv1d(384, 512, kernel_size=4)
        self.threshold2 = nn.Threshold(1e-6, 0)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        # (B, 512, 21)
        self.conv3 = nn.Conv1d(512, rnn_dim, kernel_size=4)
        self.threshold3 = nn.Threshold(1e-6, 0)
        if use_maxpool3:
            self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2)
        # (B, rnn_dim, rnn_num_steps)
        self.rnn = rnn(num_steps=rnn_num_steps, emb_dim=rnn_dim)
        # (B, rnn_dim)
        self.emb_proj = nn.Linear(rnn_dim, 1024)
        # (B, 1024)

    def forward(self, txt):
        # temporal convolutions
        out = self.conv1(txt)
        out = self.threshold1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.threshold2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.threshold3(out)
        if self.use_maxpool3:
            out = self.maxpool3(out)

        # recurrent computation
        out = out.permute(0, 2, 1)
        out = self.rnn(out)

        # linear projection
        out = self.emb_proj(out)

        return out


def str_to_labelvec(string, max_str_len=201):
    string = string.lower()
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
    # {'char': num, ...}
    alpha_to_num = {k: v + 1 for k, v in zip(alphabet, range(len(alphabet)))}
    labels = torch.zeros(max_str_len, requires_grad=False).long()
    max_i = min(max_str_len, len(string))
    for i in range(max_i):
        # Append ' ' number if char not found
        labels[i] = alpha_to_num.get(string[i], alpha_to_num[' '])
    return labels


def labelvec_to_onehot(labels):
    labels = torch.LongTensor(labels).unsqueeze(1)
    one_hot = torch.zeros(labels.size(0), 71, requires_grad=False).scatter_(1, labels, 1.)
    # Ignore zeros in one-hot mask (position 0 = empty one-hot)model.state_dict()
    one_hot = one_hot[:, 1:]
    one_hot = one_hot.permute(1, 0)
    return one_hot


def prepare_text(string, max_str_len=201):
    # Converts a text description from string format to one-hot tensor format.
    labels = str_to_labelvec(string, max_str_len)
    one_hot = labelvec_to_onehot(labels)
    return one_hot
