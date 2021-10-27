import torch as th
import math
import numpy as np
from video_loader import VideoLoader
from torch.utils.data import DataLoader
import argparse
from model import get_model_p
from preprocessing import Preprocessing
from random_sequence_shuffler import RandomSequenceSampler
import torch.nn.functional as F
import ffmpeg

# parameter setting
framerate=1
size=112
centercrop=False
type="2d" # 3d
batch_size=4
l2_normalize=1
half_precision=1
resnext101_model_path="model/resnext101.pth"
num_decoding_thread=4

def _get_video_dim( video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams']
                         if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    return height, width

def _get_output_dim( h, w):
    if isinstance(size, tuple) and len(size) == 2:
        return size
    elif h >= w:
        return int(h * size / w), size
    else:
        return size, int(w * size / h)

def load_video(video_path):
    try:
        # video_path = ""
        output_file = ""
        h, w = _get_video_dim(video_path)
    except:
        print('ffprobe failed at: {}'.format(video_path))

    height, width = _get_output_dim(h, w)
    cmd = (
        ffmpeg
            .input(video_path)
            .filter('fps', fps=framerate)
            .filter('scale', width, height)
    )
    if centercrop:
        x = int((width - size) / 2.0)
        y = int((height - size) / 2.0)
        cmd = cmd.crop(x, y, size, size)
    out, _ = (
        cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
    )
    if centercrop and isinstance(size, int):
        height, width = size, size
    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    video = th.from_numpy(video.astype('float32'))
    video = video.permute(0, 3, 1, 2)
    return video

if __name__=="__main__":
    video_file = "../dataset/videos/a.mp4"
    output_file = "../dataset/output/single_features.npy"

    preprocess = Preprocessing(type)
    model = get_model_p(type,resnext101_model_path)

    with th.no_grad():
        data = load_video(video_file)

        video = data.squeeze()
        print("len(video.shape)=", len(video.shape))

        video = preprocess(video)

        n_chunk = len(video)
        print("n_chunk = ", n_chunk)
        print("batch size = ", batch_size)
        features = th.cuda.FloatTensor(n_chunk, 2048).fill_(0)
        n_iter = int(math.ceil(n_chunk / float(batch_size)))
        print("number of inter. = ", n_iter)
        for i in range(n_iter):
            min_ind = i * batch_size
            max_ind = (i + 1) * batch_size
            video_batch = video[min_ind:max_ind].cuda()
            batch_features = model(video_batch)
            if l2_normalize:
                batch_features = F.normalize(batch_features, dim=1)
            features[min_ind:max_ind] = batch_features
        features = features.cpu().numpy()
        print("features.shape = ", features.shape)
        if half_precision:
            features = features.astype('float16')
        print("save...", output_file)
        print(features.shape)
        np.save(output_file, features)
        print()

