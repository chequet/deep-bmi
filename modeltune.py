### architecture gridsearch with ray tune
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
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

def cross_validation(data, k=5):
    # leave for now

def load_data(data_directory, type=3):
    train_files, val_files = train_val_split(data_directory + '/train/')
    data_directory = "../old_data/" + str(n_inputs) + "_data/"
    train_files, val_files = train_val_split(data_directory + '/train/',n_train=100)

    trainparams = {'batch_size': None,
                   'num_workers': 4}
    valparams = {'batch_size': None,
                 'num_workers': 2}
    if type == 1:
    elif type ==2:
    elif type == 3:
        train_iterator = iter(
            torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory + '/train/', train_files, True, 1),
                                        **trainparams))
        valid_iterator = iter(
            torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory + '/train/', val_files, True, 1),
                                        **valparams))



