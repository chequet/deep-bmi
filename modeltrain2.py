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
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

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
    for i in range(len(train_set)):
        print("batch index %i" % i, end='\r')
        batch = next(train_iterator)
        X = batch[0].to(device)
        Y = batch[1].to(device)
        model.train()
        # forward pass
        y_pred = model(X.float())
        # compute and print loss
        loss = loss_fn(y_pred, Y)
        # backward pass
        loss.backward()
        # update weights with gradient descent
        optimiser.step()
    return loss.item()

def validate(model, validation_set, validation_iterator, loss_fn, optimiser):
    with torch.no_grad():
        val_loss = 0.0
        val_r2 = 0.0
        val_r = 0.0
        for i in range(len(validation_set)):
            print("validation batch index %i" % i, end='\r')
            batch = next(validation_iterator)
            X = batch[0].to(device)
            Y = batch[1].to(device)
            model.eval()
            # forward pass
            y_pred = model(X.float())
            # compute loss
            loss = loss_fn(y_pred, Y)
            val_loss += loss.cpu().numpy()
            val_r2 += r2_score(y_pred.cpu().numpy(), Y.cpu().numpy())
            val_r += pearsonr(y_pred.ravel().cpu().numpy(), Y.ravel().cpu().numpy())[0]
        if not np.isnan(val_r):
            r = (val_r / i)
        else:
            r = 0
        r2 = (val_r2 / i)
        loss = (val_loss / i)
        return loss, r, r2

def main():

    # get or create model
    if os.path.exists(PATH):
        print("loading saved model...")
        model = torch.load(PATH)
    else:
        arch = make_architecture(N_INPUTS, 1, REDUCTIONS)
        model = FlexibleNet(arch, 0, 'ELU')
    model = model.to(device)
    print(model)
    # let's just go for elu and radam and huber loss
    #-------------PARAMS-----------------------------------------------
    learning_rate = 1e-4
    loss_fn = nn.HuberLoss()
    optimiser = torch.optim.RAdam(model.parameters(), lr=learning_rate)
    data_directory = "/data/" + str(N_SNPS) + "_data/"
    #------------------------------------------------------------------

    # 5-fold cross validation
    cross_val_partitions = k_fold_split(data_directory)
    for val_set in cross_val_partitions:
        train_set = train_val_split(val_set)
        train_iterator = get_dataloader(data_directory,ENCODING,4,train_set)
        valid_iterator = get_dataloader(data_directory,ENCODING,2,val_set)
        # initialise summary writer for tensorboard
        writer = SummaryWriter()
        # initialise early stopping
        tolerance = 10
        no_improvement = 0
        best_val_loss = np.inf
        for epoch in N_EPOCHS:
            loss = train(model,train_set,train_iterator,loss_fn,optimiser)
            # log training loss w tensorboard
            writer.add_scalar("Loss/train", loss, epoch)
            val_loss, val_r, val_r2 = validate(model,val_set,valid_iterator,loss_fn,optimiser)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Pearson_R", val_r, epoch)
            writer.add_scalar("R2", val_r2, epoch)
            # check conditions for early stopping
            if val_loss < best_val_loss:
                no_improvement = 0
                best_val_loss = val_loss
            else:
                no_improvement += 1
            if t > 5 and no_improvement == tolerance:
                print("min validation loss: %f" % best_val_loss)
                print("loss increasing for %i epochs" % no_improvement)
                print("STOPPING EARLY")
                break
        writer.flush()
        writer.close()




if __name__ == "__main__":
    main()
