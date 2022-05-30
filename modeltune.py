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
    if type == 1:
    elif type ==2:
    elif type == 3:



