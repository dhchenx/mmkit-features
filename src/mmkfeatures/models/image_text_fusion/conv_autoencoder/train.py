import torch
torch.cuda.set_per_process_memory_fraction(1.0, 0)
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from mmkfeatures.models.image_text_fusion.conv_autoencoder.models import ConvAutoencoder, ImgDataset
import argparse
import time
import os


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def time_msec():
    return int(round(time.time() * 1000))


def train(input_folder, output_folder, lr=0.001, epochs=40, img_format='.jpeg'):
    ensure_folder(output_folder)
    model = ConvAutoencoder()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    transform = transforms.ToTensor()
    train_folder = input_folder + '/train'
    print(f"Create train dataset from - {train_folder}")
    train_data = ImgDataset(train_folder, transform, device, img_format)
    print(f"Train dataset size - {len(train_data)}")
    # Create training
    num_workers = 0
    # how many samples per batch to load
    batch_size = 2
    # prepare data loaders
    print(f"Create dataloader batches={batch_size} and workers={num_workers}")
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    print("trainloader's len:",len(train_loader))
    # specify loss function
    criterion = nn.BCELoss()
    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # number of epochs to train the model
    n_epochs = epochs

    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        train_loss = 0.0

        ###################
        # Train the model #
        ###################
        model.train()
        train_loss = 0
        for images in train_loader:
            # _ stands in for labels, here
            # no need to flatten enc_64x64_images
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs, encoder_output = model(images, encoder_mode=False)
            # calculate the loss
            loss = criterion(outputs, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * images.size(0)

        # print avg training statistics
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    ###################
    # Test the model #
    ###################
    model.eval()
    test_folder = input_folder + '/test'
    test_data = ImgDataset(test_folder, transform, device, img_format)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    eval_loss = 0
    for images in test_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs, x_encoders = model(images, encoder_mode=False)
        # calculate the loss
        loss = criterion(outputs, images)
        # update running training loss
        eval_loss += loss.item() * images.size(0)

    # print avg evaluation statistics
    eval_loss = eval_loss / len(test_loader)
    print('\tEvaluation Loss: {:.6f}'.format(eval_loss))

    # Save model
    # model_root = output_folder + '/conv_autoencoder_' + str(time_msec()) + '.pt'
    model_root = output_folder + '/conv_autoencoder.pt'
    print(f"Saving trained model to: {model_root}")
    stat_dict=model.state_dict()
    torch.save(stat_dict, model_root)


def main():
    '''
train.py
../../dataset
../../output
0.001
100
    '''
    parser = argparse.ArgumentParser(description='Train conv autoencoder on dataset.'
                                                 'Must have train and test dataset folders'
                                                 ' located in input')
    parser.add_argument('input', metavar='input', type=str,
                        help='Source folder full path.')
    parser.add_argument('output', metavar='output', type=str,
                        help='Destination folder full path.')
    parser.add_argument('lr', metavar='lt', type=str,
                        help='Learning rate')
    parser.add_argument('epochs', metavar='epochs', type=str,
                        help='Number of training epochs.')

    args = parser.parse_args()
    input_folder = args.input
    output_folder = args.output
    lr = args.lr
    epochs = args.epochs

    train(input_folder, output_folder, float(lr), int(epochs))


if __name__ == '__main__':
    print("<<<<<<<<<<<<<<< Start training >>>>>>>>>>>>>>>>>")
    main()
