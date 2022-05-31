### architecture gridsearch with ray tune
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from MyIterableDataset3 import *
from OneHotIterableDataset import *
from BasicEmbeddedDataset import *
from EffectEmbeddingDataset import *
import os, re, sys
from modeltrain import train_val_split
import FlexibleNet

N_INPUTS = 1000
N_EPOCHS = 200

def cross_validation(data, k=5):
    # leave for now

def get_dataloaders(data_directory, type=3):
    train_files, val_files = train_val_split(data_directory + '/train/',n_train=100)
    trainparams = {'batch_size': None,
                   'num_workers': 4}
    valparams = {'batch_size': None,
                 'num_workers': 2}
    #TODO add code for other encodings
    if type == 1:
        # code for SVE
    elif type ==2:
        # code for one hot
    elif type == 3:
        train_iterator = iter(
            torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory + '/train/', train_files, True, 1),
                                        **trainparams))
        valid_iterator = iter(
            torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory + '/train/', val_files, True, 1),
                                        **valparams))
    return train_iterator, valid_iterator

def train(config, checkpoint_dir=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    # set up model according to config
    model = FlexibleNet(config['arch'],config['dropout'],config['activation'])
    loss_fn = torch.nn.MSELoss(reduction='mean')
    learning_rate = config['lr']
    # choose optimiser based on config
    if config['optim'] == 'adam':
        optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    # TODO add other optimiser options!
    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimiser.load_state_dict(optimizer_state)
    data_directory = "../old_data/" + str(N_INPUTS) + "_data/"
    train_iterator, valid_iterator = get_dataloaders(data_directory, type=config['enc'])
    # train
    i = 0
    acc = 0
    while i < n_trainbatch:
        print("batch index %i" % i, end='\r')
        batch = next(batch_iterator)
        X = batch[0].to(device)
        # if clf:
            # X = X.reshape(X.shape[0],X.shape[1],1)
            # print(X)
        Y = batch[1].to(device)
        model.train()
        #X = X.short()
        # forward pass
        y_pred = model(X.float())
        if clf:
            y_round = y_pred.argmax(dim=1)
            acc = binary_acc(y_round.cpu(), Y.flatten().cpu())
            Y = Y.flatten().long()
        # compute and print loss
        loss = loss_fn(y_pred, Y)
        # Zero the gradients before running the backward pass.
        optimiser.zero_grad()
        # backward pass
        loss.backward()
        # update weights with gradient descent
        optimiser.step()
        i += 1




