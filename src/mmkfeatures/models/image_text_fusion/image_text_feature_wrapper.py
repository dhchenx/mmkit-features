import mmkfeatures.models.image_text_fusion.char_embedding.train as tr_char
import mmkfeatures.models.image_text_fusion.conv_autoencoder.train as tr_conv
from mmkfeatures.models.image_text_fusion.char_embedding.char_cnn_rnn import CharCnnRnn
from mmkfeatures.models.image_text_fusion.conv_autoencoder.models import ConvAutoencoder
import torch
import cv2
import ntpath
import torchvision.transforms as transforms

class ImageTextFeaturesWrapper:
    def __init__(self):
        pass

    def train_image_conv_model(self,
                                   input_folder,output_folder,lr=0.001,epochs=100
                                   ):
        tr_conv.train(input_folder, output_folder, lr, epochs)

    def train_char_embedding_model(self,
                               json_path,output_folder,lr=0.001,epochs=100,model_type="fixed_gru",rnn_type="cvpr",img_tag="img_64x64_path"
                               ):

        tr_char.train(json_path=json_path, output_path=output_folder, learning_rate=float(lr), epochs=int(epochs),
              model_type=model_type, rnn_type=rnn_type, img_tag=img_tag)

    def prepare_text(self,string, max_str_len=201):
        # Converts a text description from string format to one-hot tensor format.
        labels = self.str_to_labelvec(string, max_str_len)
        one_hot = self.labelvec_to_onehot(labels)
        return one_hot

    def labelvec_to_onehot(self,labels):
        labels = torch.LongTensor(labels).unsqueeze(1)
        one_hot = torch.zeros(labels.size(0), 71, requires_grad=False).scatter_(1, labels, 1.)
        # Ignore zeros in one-hot mask (position 0 = empty one-hot)model.state_dict()
        one_hot = one_hot[:, 1:]
        one_hot = one_hot.permute(1, 0)
        return one_hot

    def str_to_labelvec(self,string, max_str_len=201):
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

    def get_embedding_text(self,embedding_path,text,output_path=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embedding = CharCnnRnn()
        embedding.load_state_dict(torch.load(embedding_path))
        embedding = embedding.to(device)
        embeddings=[]
        torch.no_grad()
        embedding.eval()
        text = self.prepare_text(text)
        text = text.to(device)
        embedded_txt = embedding(text.unsqueeze(0))
        if output_path!=None :
            torch.save(embedded_txt, output_path)
        return embedded_txt

    def get_encode_image(img, encoder_path,image_path, output_file=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = ConvAutoencoder()
        encoder.load_state_dict(torch.load(encoder_path))
        encoder = encoder.to(device)
        torch.no_grad()
        encoder.eval()
        transform = transforms.ToTensor()
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = transform(image)
        image = image.float()
        image = image.to(device)
        enc_img = encoder(image.unsqueeze(0), encoder_mode=True)
        if output_file!=None:
            torch.save(enc_img, output_file)
        return enc_img







