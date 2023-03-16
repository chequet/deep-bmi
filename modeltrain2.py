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
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
#import pickle
import csv
import sys



# PARAMS TO CHANGE ============================
N_SNPS = int(sys.argv[1])
N_INPUTS = int(sys.argv[2])
ENCODING = int(sys.argv[3])
N_EPOCHS = int(sys.argv[4])
BATCH_SIZE = 4096
REDUCTIONS = [2,2,5,5]
PATH = "NEW_" + str(N_SNPS) + '_huber_relu_0_adamax_' + str(ENCODING) + ".pt"
#==============================================

def make_architecture(inp, outp, reduction_factors):
    arch = [inp]
    current = inp
    for i in range(len(reduction_factors)):
        redf = reduction_factors[i]
        next_layer = math.ceil(current/redf)
        arch.append(next_layer)
        current = next_layer
    arch.append(outp)
    return arch

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
    # check no overlap
    assert( not [file for file in partitions[0] if file in partitions[1]] )
    return partitions

def get_train_files(train_dir, val_files):
    # get training set given holdout set
    files = os.listdir(train_dir)
    train_files = [item for item in files if item not in val_files]
    return train_files

def get_dataloader(data_directory, encoding, workers, files):
    params = {'batch_size': None,
              'num_workers': workers}
    dataloader = None
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
    i = 0
    while i < len(train_set):
        print("batch index %i" % i, end='\r')
        batch = next(train_iterator)
        X = batch[0].to(device)
        Y = batch[1].to(device)
        optimiser.zero_grad()
        model.train()
        # forward pass
        y_pred = model(X.float())
        # compute and print loss
        loss = loss_fn(y_pred, Y)
        # backward pass
        loss.backward()
        # update weights with gradient descent
        optimiser.step()
        i += 1
    return loss.item()

def validate(model, validation_set, validation_iterator, loss_fn, optimiser):
    with torch.no_grad():
        val_loss = 0.0
        val_r2 = 0.0
        val_r = 0.0
        i = 0
        while i < len(validation_set):
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
            i+=1
        if not np.isnan(val_r):
            r = (val_r / i)
        else:
            r = 0
        r2 = (val_r2 / i)
        loss = (val_loss / i)
        return loss, r, r2

def train_and_validate(arch, data_directory, train_set, val_set):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    # new model
    # -------------PARAMS-----------------------------------------------
    model = FlexibleNet(arch, 0, 'ReLU').to(device)
    learning_rate = 0.0001
    loss_fn = nn.HuberLoss()
    #loss_fn = nn.MSELoss(reduction='mean')
    optimiser = optim.Adamax(model.parameters(), lr=learning_rate)
    # ------------------------------------------------------------------
    # # initialise summary writer for tensorboard
    writer = SummaryWriter()
    # initialise early stopping
    tolerance = 10
    no_improvement = 0
    best_val_loss = np.inf
    for t in range(N_EPOCHS):
        print("epoch %i" % t)
        train_iterator = get_dataloader(data_directory, ENCODING, 12, train_set)
        valid_iterator = get_dataloader(data_directory, ENCODING, 6, val_set)
        i = 0
        while i < len(train_set):
            print("batch index %i" % i, end='\r')
            batch = next(train_iterator)
            X = batch[0].to(device)
            Y = batch[1].to(device)
            optimiser.zero_grad()
            model.train()
            # forward pass
            y_pred = model(X.float())
            # compute and print loss
            loss = loss_fn(y_pred, Y)
            # backward pass
            loss.backward()
            # update weights with gradient descent
            optimiser.step()
            i += 1
        print("training loss: %f" % loss)
        # log training loss w tensorboard
        writer.add_scalar("Loss/train", loss, t)
        with torch.no_grad():
            val_loss = 0.0
            val_r2 = 0.0
            val_r = 0.0
            i = 0
            while i < len(val_set):
                print("validation batch index %i" % i, end='\r')
                batch = next(valid_iterator)
                X = batch[0].to(device)
                Y = batch[1].to(device)
                model.eval()
                # forward pass
                y_pred = model(X.float())
                print(Y[:10])
                print(y_pred[:10])
                # compute loss
                loss = loss_fn(y_pred, Y)
                val_loss += loss.cpu().numpy()
                val_r2 += r2_score(y_pred.cpu().numpy(), Y.cpu().numpy())
                val_r += pearsonr(y_pred.ravel().cpu().numpy(), Y.ravel().cpu().numpy())[0]
                i += 1
            if not np.isnan(val_r):
                r = (val_r / i)
            else:
                r = 0
            r2 = (val_r2 / i)
            loss = (val_loss / i)
        print("validation loss: %f" % loss)
        print("pearson r: %f" % r)
        writer.add_scalar("Loss/val", loss, t)
        writer.add_scalar("Pearson_R", r, t)
        writer.add_scalar("R2", r2, t)
        # check conditions for early stopping
        if t > 20:
            if t % 10 == 0:
                print("no improvement for %i epochs" % no_improvement)
            if loss < best_val_loss:
                no_improvement = 0
                best_val_loss = loss
            else:
                no_improvement += 1
            if no_improvement >= tolerance:
                print("best validation loss: %f" % best_val_loss)
                print("STOPPING EARLY\n\n")
                break
    writer.flush()
    writer.close()
    torch.save(model, PATH)
    return loss, r, r2, t


def main():
    arch = make_architecture(N_INPUTS, 1, REDUCTIONS)
    # save results for printing and persisting
    results = {'validation_sets':[], 'validation_loss':[], 'validation_r':[], 'validation_r2':[], 'n_epochs':[]}
    # 5-fold cross validation
    data_directory = "/data/" + str(N_SNPS) + "_data_relabelled/"
    cross_val_partitions = k_fold_split(data_directory+'train/')

    for val_set in cross_val_partitions:
        results['validation_sets'].append(val_set)
        train_set = get_train_files(data_directory+'train/', val_set)
        print("\nvalidation set:")
        print(val_set)
        val_loss, val_r, val_r2, t = train_and_validate(arch, data_directory, train_set, val_set)
        results['validation_loss'].append(val_loss)
        results['validation_r'].append(val_r)
        results['validation_r2'].append(val_r2)
        results['n_epochs'].append(t)
    # save results
    results_path = '../results/' + PATH + '_results.csv'
    with open(results_path, 'w') as f:
        w = csv.writer(f)
        w.writerows(results.items())
    # print interesting results
    print('mean validation loss: %f' %np.mean(np.array(results['validation_loss'])))
    print('best validation loss: %f' % np.min(np.array(results['validation_loss'])))
    print('mean validation r: %f' % np.mean(np.array(results['validation_r'])))
    print('best validation r: %f' % np.max(np.array(results['validation_r'])))
    print('mean validation r2: %f' % np.mean(np.array(results['validation_r2'])))
    print('best validation r2: %f' % np.max(np.array(results['validation_r2'])))
    print('mean number of epochs: %i' %np.mean(np.array(results['n_epochs'])))
    print('epoch range: %i' %( np.max(np.array(results['n_epochs'])) - np.min(np.array(results['n_epochs'])) ))

if __name__ == "__main__":
    main()
