"""Main pipeline of DA-RNN.

@author Zhenye Na 05/21/2018

"""

from random import seed
from random import randint

import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# from tqdm import tqdm
from torch.autograd import Variable


from core.autoencoders.simple_autoencoder import Autoencoder
from core.autoencoders.cnn_autoencoder import AutoencoderCNN

from core.data.data_loader import TimeSeriesData

from torch.utils.data import DataLoader

#from ops import *
from model import *


def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="PyTorch implementation of paper 'A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction'")

    # Dataset setting
    parser.add_argument('--dataroot', type=str, default="../data/nasdaq100_padding-short.csv", help='path to dataset')
    #parser.add_argument('--dataroot', type=str, default="../data/BTC-USD-4H-24F.csv", help='path to dataset')
    
    parser.add_argument('--batchsize', type=int, default=128, help='input batch size [128]')

    # Encoder / Decoder parameters setting
    parser.add_argument('--nhidden_encoder', type=int, default=128, help='size of hidden states for the encoder m [64, 128]')
    parser.add_argument('--nhidden_decoder', type=int, default=128, help='size of hidden states for the decoder p [64, 128]')
    parser.add_argument('--ntimestep', type=int, default=10, help='the number of time steps in the window T [10]')

    # Training parameters setting
    parser.add_argument('--epochs', type=int, default=70, help='number of epochs to train [10, 200, 500]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')

    # parse the arguments
    args = parser.parse_args()

    return args


def main():
    """Main pipeline of DA-RNN."""
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cuda'
    print('==> Using device:', device)

    # Read dataset
    print("==> Load dataset ...")
    
    num_epochs = 20
    batch_size = 32

    Normalise =  True
    n_features = 24
    intermediate = 8
    idx_class = 1
    batch_size = 128

    #X, y = read_data(args.dataroot, normalise=True, debug=False)
    #X, y = load_data_series('../data/BTC-USD-4H-24F.csv', n_features, idx_class, Normalise)
    #y = y.reshape(-1,)

    #print(X.shape)
    #print(y.shape)


    dataset = TimeSeriesData('../data/BTC-USD-4H-24F.csv', n_features, idx_class, Normalise)
    X, y = dataset.load_data_series(0)
    y = y.reshape(-1,)
    print(X.shape)
    print(y.shape)

    #dataset_train = TimeSeriesData('../data/BTC-USD-4H-24F-train.csv', n_features, idx_class, Normalise)
    #dataset_test = TimeSeriesData('../data/BTC-USD-4H-24F-test.csv', n_features, idx_class, Normalise)
    
    # cargamos os loaders
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)


    loadModel = True
    path_saved = '../saved/simple_autoencoder.pth'

    autoencoder = Autoencoder(device, num_epochs, n_features, intermediate).double().to(device)

    if loadModel:
        print('[Model] loading pre-training values ...')
        #device = 'cpu'
        autoencoder = Autoencoder(device, num_epochs, n_features, intermediate).to(device).load_checkpoint(path_saved)
    else:
        print('training....')
        autoencoder.train(train_loader)

    predicted = autoencoder.test(train_loader)
    predicted = np.array(predicted)
    print(len(predicted))
    X = predicted
    print(X.shape)
    print(y.shape)


    # Initialize model
    print("==> Initialize DA-RNN model ...")

    loadModel = False
    model = DA_rnn(
            X,
            y,
            args.ntimestep,
            args.nhidden_encoder,
            args.nhidden_decoder,
            args.batchsize,
            args.lr,
            args.epochs
        )

    if loadModel:
        print('loading model ...')
        model, opt = model.load_checkpoint('../saved/checkpoint.pth')
        print(model)
    
    # Train
    print("==> Start training ...")
    model.train()

    # Prediction
    y_pred = model.test()
    model.save_model('../saved/checkpoint.pth')
    print(model)
    
    fig1 = plt.figure()
    plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
    plt.savefig("1.png")
    plt.close(fig1)

    fig2 = plt.figure()
    plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
    plt.savefig("2.png")
    plt.close(fig2)

    fig3 = plt.figure()
    plt.plot(y_pred, label='Predicted')
    plt.plot(model.y[model.train_timesteps:], label="True")
    plt.legend(loc='upper left')
    plt.savefig("3.png")
    plt.close(fig3)
    print('Finished Training')


if __name__ == '__main__':
    main()
    #pass
