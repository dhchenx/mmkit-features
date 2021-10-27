import torch as th
import math
import numpy as np
from mmkfeatures.video.model import get_model_p
from mmkfeatures.video.preprocessing import Preprocessing
import torch.nn.functional as F
import ffmpeg

class VideoFeatureWrapper:
    def __init__(self):
        self.framerate = 1
        self.size = 112
        self.centercrop = False
        self.model_type = "2d"  # 3d
        self.batch_size = 4
        self.l2_normalize = 1
        self.half_precision = 1
        self.resnext101_model_path = "model/resnext101.pth"
        self.num_decoding_thread = 4

    def _get_video_dim(self,video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        return height, width

    def _get_output_dim(self,h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def load_video(self,video_path):
        try:
            # video_path = ""
            output_file = ""
            h, w = self._get_video_dim(video_path)
        except:
            print('ffprobe failed at: {}'.format(video_path))

        height, width = self._get_output_dim(h, w)
        cmd = (
            ffmpeg
                .input(video_path)
                .filter('fps', fps=self.framerate)
                .filter('scale', width, height)
        )
        if self.centercrop:
            x = int((width - self.size) / 2.0)
            y = int((height - self.size) / 2.0)
            cmd = cmd.crop(x, y, self.size, self.size)
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
        )
        if self.centercrop and isinstance(self.size, int):
            height, width = self.size, self.size
        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        video = th.from_numpy(video.astype('float32'))
        video = video.permute(0, 3, 1, 2)
        return video

    def extract_video_features(self,video_file,output_file=None):
        #video_file = "../dataset/videos/a.mp4"
        #output_file = "../dataset/output/single_features.npy"

        preprocess = Preprocessing(self.model_type)
        model = get_model_p(self.model_type, self.resnext101_model_path)

        with th.no_grad():
            data = self.load_video(video_file)

            video = data.squeeze()
            print("len(video.shape)=", len(video.shape))

            video = preprocess(video)

            n_chunk = len(video)
            print("n_chunk = ", n_chunk)
            print("batch size = ", self.batch_size)
            features = th.cuda.FloatTensor(n_chunk, 2048).fill_(0)
            n_iter = int(math.ceil(n_chunk / float(self.batch_size)))
            print("number of inter. = ", n_iter)
            for i in range(n_iter):
                min_ind = i * self.batch_size
                max_ind = (i + 1) * self.batch_size
                video_batch = video[min_ind:max_ind].cuda()
                batch_features = model(video_batch)
                if self.l2_normalize:
                    batch_features = F.normalize(batch_features, dim=1)
                features[min_ind:max_ind] = batch_features
            features = features.cpu().numpy()
            print("features.shape = ", features.shape)
            if self.half_precision:
                features = features.astype('float16')
            print("save...", output_file)
            # print(features.shape)
            if output_file!=None:
                np.save(output_file, features)
            print()
        return features