import json
import torch

torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(1.0, 0)
torch.cuda.memory_summary(device=None, abbreviated=False)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from torch.utils.data import DataLoader
import random
import os
import argparse
import time
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import models
path = os.path.dirname(os.path.abspath(__file__))
path = Path(path)
src_path = path.parent
print("Add models path to sys paths - ", src_path)
sys.path.append(str(src_path) + '/models/char_embedding/')
from mmkfeatures.models.image_text_fusion.char_embedding.char_cnn_rnn import CharCnnRnn
from mmkfeatures.models.image_text_fusion.char_embedding.multi_model_dataset import MultimodalDataset

def train(json_path, output_path, learning_rate=0.001, img_tag='encod_64x64_path',
          learning_rate_decay=0.1, batch_size=20, epochs=20,
          symmetric=True, num_workers=0, model_type='fixed_gru', rnn_type='cvpr'):
    ensure_folder(output_path)
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create datasets loader
    train_data, test_data = get_datasets(json_path, device, 0.8, img_tag)
    print("Creating dataloaders ...")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    print("Dataloaders created!")
    # create Char-CNN-RNN model
    model = CharCnnRnn(model_type, rnn_type)
    print(f"CharCnnRnn created device:{device}")
    optim = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    # Adjust lr in progress
    sched = torch.optim.lr_scheduler.ExponentialLR(optim, learning_rate_decay)

    # torch.cuda.empty_cache()
    # torch.set_grad_enabled(False)

    model.to(device)
    model.train()
    ###################
    # train the model #
    ###################

    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            img = data['img']
            txt = data['text']
            feat_img = img.view(img.size(0), -1)
            feat_txt = model(txt)
            loss1, acc1 = sje_loss(feat_txt, feat_img)
            loss2 = acc2 = 0
            if symmetric:
                loss2, acc2 = sje_loss(feat_img, feat_txt)
            loss = loss1 + loss2
            model.zero_grad()
            loss.backward()
            optim.step()


        print(f"Epoch: {epoch} Loss: {loss}")
        sched.step()
    ######################
    # evaluate the model #
    ######################
    print("evaling...")
    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)
    model.eval()
    eval_loss = 0
    for data in test_loader:

        img = data['img']
        txt = data['text']
        feat_img = img.view(img.size(0), -1)
        feat_txt = model(txt)
        loss1, acc1 = sje_loss(feat_txt, feat_img)
        loss2 = acc2 = 0
        if symmetric:
            loss2, acc2 = sje_loss(feat_img, feat_txt)
        loss = loss1 + loss2
        # update evaluation loss
        eval_loss += loss
    eval_loss = eval_loss / len(test_loader)
    print(f"Evaluation loss: {eval_loss}")
    # Save model
    # model_root = output_path + '/char_embedding' + str(time_msec()) + '.pt'
    model_root = output_path + '/char_embedding.pt'
    print(f"Saving trained model to: {model_root}")
    # torch.cuda.empty_cache()

    model_state_dict=model.state_dict()
    torch.save(model_state_dict, model_root)


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def time_msec():
    return int(round(time.time() * 1000))


def get_datasets(dataset_json_path, device, split=0.8, img_tag='encod_64x64_path'):
    # Load data and create train and test datasets loaders
    data_folder = os.path.dirname(dataset_json_path)
    items = []
    with open(dataset_json_path) as f:
        dataset_json = json.load(f)
        # dataset_json=random.sample(dataset_json,60000)
        print(f"Load dataset size: {len(dataset_json)}")
        for item in dataset_json:
            items.append({'img': item[img_tag], 'text': item['text']})
        # Split train and test datasets
        train_size = int(len(items) * split)
        train_indx = random.sample(range(len(items)), k=train_size)
        indx = [i for i in range(len(items))]
        test_indx = list(set(indx) - set(train_indx))
        train_dataset = [items[i] for i in train_indx]
        test_dataset = [items[i] for i in test_indx]
        print(f"Train : {len(train_dataset)} Test: {len(test_dataset)}")

        print(f"Creating MultimodalDatasets from: {data_folder}")
        train_data = MultimodalDataset(train_dataset, data_folder, device)
        test_data = MultimodalDataset(test_dataset, data_folder, device)

        print("MultimodalDatasets created!")

        return train_data, test_data


def sje_loss(feat1, feat2):
    # Structured Joint Embedding Loss
    # similarity score matrix (rows: fixed feat2, columns: fixed feat1)
    # print(feat2.shape,feat1.t().shape)
    scores = torch.matmul(feat2, feat1.t())  # (B, B)
    # diagonal: matching pairs
    diagonal = scores.diag().view(scores.size(0), 1)  # (B, 1)
    # repeat diagonal scores on rows
    diagonal = diagonal.expand_as(scores)  # (B, B)
    # calculate costs
    cost = (1 + scores - diagonal).clamp(min=0)  # (B, B)
    # clear diagonals (matching pairs are not used in loss computation)
    # cost[torch.eye(cost.size(0)).bool()] = 0 # (B, B) for torch==1.2.0
    cost[torch.eye(cost.size(0), dtype=torch.bool)] = 0  # (B, B)
    # sum and average costs
    denom = cost.size(0) * cost.size(1)
    loss = cost.sum() / denom

    # batch accuracy
    max_ids = torch.argmax(scores, dim=1)
    ground_truths = torch.LongTensor(range(scores.size(0))).to(feat1.device)
    num_correct = (max_ids == ground_truths).sum().float()
    accuracy = 100 * num_correct / cost.size(0)

    return loss, accuracy


def main():
    '''
train.py
../../dataset/images_data.json
../../output
0.001
100
fixed_gru
cvpr
encod_64x64_path
    '''
    parser = argparse.ArgumentParser(description='Train embedding model on dataset.')
    parser.add_argument('json_path', metavar='json_path', type=str,
                        help='Source folder full path.')
    parser.add_argument('output', metavar='output', type=str,
                        help='Destination folder full path.')
    parser.add_argument('lr', metavar='lt', type=str,
                        help='Learning rate')
    parser.add_argument('epochs', metavar='epochs', type=str,
                        help='Number of training epochs.')
    parser.add_argument('model_type', metavar='model_type', type=str,
                        help='def fixed_gru|fixed_rnn')
    parser.add_argument('rnn_type', metavar='rnn_type', type=str,
                        help='def cvpr|icml')
    parser.add_argument('img_tag', metavar='img_tag', type=str,
                        help='def img_64x64_path|img_256x256_path')

    args = parser.parse_args()
    json_path = args.json_path
    output_folder = args.output
    lr = args.lr
    epochs = args.epochs
    model_type = args.model_type
    rnn_type = args.rnn_type
    img_tag = args.img_tag
    train(json_path=json_path, output_path=output_folder, learning_rate=float(lr), epochs=int(epochs), model_type=model_type, rnn_type=rnn_type, img_tag=img_tag)


if __name__ == '__main__':
    print("<<<<<<<<<<<<<<< Start training >>>>>>>>>>>>>>>>>")
    main()
