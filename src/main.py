"""Main pipeline of DA-RNN.

@author Zhenye Na 05/21/2018
@modified Steve Ataucuri 

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
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset

from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

#from ops import *
from model import *
from core.autoencoders.simple_autoencoder import Autoencoder
from core.autoencoders.cnn_autoencoder import AutoencoderCNN
from core.data.data_loader import TimeSeriesData, KindNormalization

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="PyTorch implementation of paper 'A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction'")

    # Dataset setting
    parser.add_argument('--dataroot', type=str, default="../data/nasdaq100_padding-short.csv", help='path to dataset')
    #parser.add_argument('--dataroot', type=str, default="../data/BTC-USD-4H-24F.csv", help='path to dataset')
    
    parser.add_argument('--batchsize', type=int, default=64, help='input batch size [128]')

    # Encoder / Decoder parameters setting
    parser.add_argument('--nhidden_encoder', type=int, default=64, help='size of hidden states for the encoder m [64, 128]')
    parser.add_argument('--nhidden_decoder', type=int, default=64, help='size of hidden states for the decoder p [64, 128]')
    parser.add_argument('--ntimestep', type=int, default=10, help='the number of time steps in the window T [10]')

    # Training parameters setting
    parser.add_argument('--epochs', type=int, default=70, help='number of epochs to train [10, 200, 500]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')
    parser.add_argument('--train_size', type=float, default=0.7, help='train_split')

    # parse the arguments
    args = parser.parse_args()

    return args

def calc_score(y_true, y_predicted, report=False):

    r2 = r2_score(y_true, y_predicted)
    mse = mean_squared_error(y_true, y_predicted)
    rmse = sqrt(mean_squared_error(y_true, y_predicted))
    mae = mean_absolute_error(y_true, y_predicted)

    report_string = ""
    report_string += "---Regression Scores--- \n"
    report_string += "\tR_2 statistics        (R2)  = " + str(round(r2,3)) + "\n"
    report_string += "\tMean Square Error     (MSE) = " + str(round(mse,3)) + "\n"
    report_string += "\tRoot Mean Square Error(RMSE) = " + str(round(rmse,3)) + "\n"
    report_string += "\tMean Absolute Error   (MAE) = " + str(round(mae,3)) + "\n"

    if report:
        return r2, mse, rmse, mae, report_string
    else:
        return r2, mse, rmse, mae


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

    args_autoenc = {
        'seed':7070,
        'batchsize': 128,
        'lr': 0.001,
        'epochs': 20,
        'features': 25,        # numero de caracteristicas na entrada
        'nhidden_encoder': 8,  # numero de caracteristicas da camada intermedia do Autoencoder
        'idx_class': 1,        # indice da classe a aprender nos dados neste caso 1 ou seja High
        'normalise': True
    }  

    # Reading data
    print("==> Reading Data ...")

    path_file = '../data/BTC-USD-4H-24F.csv'

    ts_data = TimeSeriesData(path_file, 
                             args_autoenc['features'], 
                             args_autoenc['idx_class'], 
                             args_autoenc['normalise'],
                             KindNormalization.Scaling)    
 
    X, y = ts_data.load_data_series(0)
    y = y.reshape(-1, 1)
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.2, random_state=42)

    print("dataset orig ", X.shape)
    print("x_train %s, y_train %s" % (x_train.shape, y_train.shape))
    print("x_test %s, y_test %s" % (x_test.shape, y_test.shape))
    print("x_val %s, y_val %s" % (x_val.shape, y_val.shape))

    # cargamos os loaders
    #train_loader = DataLoader(dataset=ts_data, batch_size=batch_size, shuffle=True)
    
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                                batch_size=args_autoenc['batchsize'], shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
                                batch_size=args_autoenc['batchsize'], shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
                                batch_size=args_autoenc['batchsize'], shuffle=True)

    dataloaders = {
        "train": train_loader,
        "validation": val_loader
    }
    
    loadModel = True
    path_saved = 'simple_autoencoder.pth'

    # Autoencoder model
    print("==> Initialize Auto-Encoder model ...")

    autoencoder = Autoencoder(device, 
                        epochs = args_autoenc['epochs'], 
                        learning_rate = args_autoenc['lr'],
                        original_dim = args_autoenc['features'],
                        intermediate_dim = args_autoenc['nhidden_encoder']        ).double().to(device)

    if loadModel:
        print('==> Loading pre-training values ...')
        #device = 'cpu'
        autoencoder = Autoencoder(device,
                        epochs = args_autoenc['epochs'], 
                        learning_rate = args_autoenc['lr'],
                        original_dim = args_autoenc['features'],
                        intermediate_dim = args_autoenc['nhidden_encoder']).to(device).load_checkpoint(path_saved)
    else:
        print('==> Training Auto-Encoder ...')
        autoencoder.train_advanced(dataloaders, show_plot=False)
        #autoencoder.train_model(train_loader)
        autoencoder.save_model(path_saved)
        
      
    #predicted = autoencoder.train_advanced(test_loader)
    encoder_layer = autoencoder.test(test_loader, 'encoder')

    encoder_layer_ = DataLoader(TensorDataset(torch.from_numpy(np.array(encoder_layer)), torch.from_numpy(y_test)),
                                batch_size=args_autoenc['batchsize'], shuffle=True)

    pred_input = autoencoder.test(encoder_layer_, 'decoder') 
    pred_input = np.array(pred_input)

    print("pred output  : ", pred_input.shape)
    print('y target     : ', y.shape)

    # metrics
    print("==> Metrics for Auto-Encoder ...")
    _,_,_,_, results = calc_score(x_test, pred_input, report=True)
    print(results)

    # Initialize model
    print("==> Initialize DA-RNN model ...")

    """default arguments for attention"""
    args = {
        'seed':7071,
        'ntimestep': 10,
        'nhidden_encoder': 128,
        'nhidden_decoder': 128,
        'batchsize': 32,
        'lr': 0.001,
        'epochs': 20,
        'train_size': 0.7
    }   

    y = y_test.reshape(-1, )
    encoder_layer = np.array(encoder_layer)    

    loadModel = False
    model = DA_rnn(
        encoder_layer,
        y,
        args["ntimestep"],
        args["nhidden_encoder"],
        args["nhidden_decoder"],
        args["batchsize"],
        args["lr"],
        args["epochs"], 
        args["train_size"]        
    )

    if loadModel:
        print('loading model ...')
        model, opt = model.load_checkpoint('checkpoint.pth')
    else:
        # Train
        print("==> Start training ...")
        print('shape dataset : %s ', encoder_layer.shape)
        model.train()


    # Prediction
    y_true = model.y[model.train_timesteps:]
    y_pred = model.test()

    print('shape y_true : %s y_pred : %s' % (y_true.shape, y_pred.shape))

    # metrics
    print("==> Metrics for Prediction ...")
    _,_,_,_, results = calc_score(y_true, y_pred, report=True)
    print(results)

    model.save_model('checkpoint.pth')

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
    plt.plot(y_true, label="True")
    plt.legend(loc='upper left')
    plt.savefig("3.png")
    plt.close(fig3)
    print('Finished Training')


if __name__ == '__main__':
    main()
    #pass
