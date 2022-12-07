### TIDIER SCRIPT FOR TRAINING MODELS
from FlexibleNet import *
from MyIterableDataset3 import *
from OneHotIterableDataset import *
from BasicEmbeddedDataset import *
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from modeltune import make_architecture

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

# PARAMS TO CHANGE ============================
N_SNPS = int(sys.argv[2])
N_INPUTS = int(sys.argv[3])
N_EPOCHS = int(sys.argv[5])
ENCODING = int(sys.argv[4])
BATCH_SIZE = 4096
REDUCTIONS = [50, 10, 10]
PATH = sys.argv[1]
#==============================================

def k_fold_split(train_dir, n=5):
    # divide training set into n groups for cross validation
    partitions = []
    files = os.listdir(train_dir)
    np.random.shuffle(files)
    partition_size = math.floor(len(files)/n)
    start = 0
    for i in range(n-1):
        stop = start + partition_size
        p = files[start:stop]
        start = stop
        partitions.append(p)
    partitions.append(files[start:])
    return partitions

def train_val_split(train_dir, val_files):
    # get training set given holdout set
    files = os.listdir(train_dir)
    train_files = [item for item in files if item not in val_files]
    return train_files

def get_dataloader(data_directory, encoding, workers, files):
    params = {'batch_size': None,
              'num_workers': workers}
    if encoding == 1:
        dataloader = iter(torch.utils.data.DataLoader
                              (MyIterableDataset(data_directory + 'train/', files, True),**params))
    elif encoding == 2:
        dataloader = iter(torch.utils.data.DataLoader
                              (OneHotIterableDataset(data_directory + 'train/', files, True),**params))
    elif encoding == 3:
        dataloader = iter(torch.utils.data.DataLoader
                              (BasicEmbeddedDataset(data_directory + 'train/', files, True, 1),**params))
    elif encoding == 4:
        dataloader = iter(torch.utils.data.DataLoader
                              (BasicEmbeddedDataset(data_directory + 'train/', files, True, 2),**params))
    return dataloader

def train(model, train_set, train_iterator, loss_fn, optimiser):


def validate(model, validation_set, validation_iterator, loss_fn, optimiser):

def main():
    # let's just go for elu and radam and huber loss
    if os.path.exists(PATH):
        print("loading saved model...")
        model = torch.load(PATH)
    else:
        arch = make_architecture(N_INPUTS, 1, REDUCTIONS)
        model = FlexibleNet(arch, 0, 'ELU')
    model = model.to(device)
    print(model)
    learning_rate = 1e-4
    loss_fn = nn.HuberLoss()
    optimiser = torch.optim.RAdam(model.parameters(), lr=learning_rate)
    data_directory = "/data/" + str(N_SNPS) + "_data/"

    # 5-fold cross validation
    cross_val_partitions = k_fold_split(data_directory)
    for val_set in cross_val_partitions:
        train_set = train_val_split(val_set)



if __name__ == "__main__":
    main()
