import matplotlib.pyplot as plt
import numpy as np
import torch
import os, re, sys
from torch import optim
from time import time
import torch.nn as nn
from scipy import stats
from MyIterableDataset3 import *
from OneHotIterableDataset import *
from BasicEmbeddedDataset import *
from EffectEmbeddingDataset import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)


def model1():
    n_features = 50000
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, 500),
        torch.nn.ELU(),
        torch.nn.Linear(500, 50),
        torch.nn.ELU(),
        torch.nn.Linear(50, 1)
    )
    return model


def onehot_model():
    n_features = 150000
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, 1500),
        torch.nn.ELU(),
        torch.nn.Linear(1500, 150),
        torch.nn.ELU(),
        torch.nn.Linear(150, 15),
        torch.nn.ELU(),
        torch.nn.Linear(15, 1)
    )
    return model


def embed_model():
    n_features = 100000
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, 1000),
        torch.nn.Dropout(0.4),
        torch.nn.ELU(),
        torch.nn.Linear(1000, 100),
        torch.nn.Dropout(0.4),
        torch.nn.ELU(),
        torch.nn.Linear(100, 10),
        torch.nn.Dropout(0.4),
        torch.nn.ELU(),
        torch.nn.Linear(10, 1)
    )
    return model

def embed_model2():
    n_features = 100000
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, 5000),
        torch.nn.Dropout(0.4),
        torch.nn.ELU(),
        torch.nn.Linear(5000, 5000),
        torch.nn.Dropout(0.4),
        torch.nn.ELU(),
        torch.nn.Linear(5000, 1000),
        torch.nn.Dropout(0.4),
        torch.nn.ELU(),
        torch.nn.Linear(1000, 1000),
        torch.nn.Dropout(0.4),
        torch.nn.ELU(),
        torch.nn.Linear(1000, 100),
        torch.nn.Dropout(0.4),
        torch.nn.ELU(),
        torch.nn.Linear(100, 100),
        torch.nn.Dropout(0.4),
        torch.nn.ELU(),
        torch.nn.Linear(100, 10),
        torch.nn.Dropout(0.4),
        torch.nn.ELU(),
        torch.nn.Linear(10, 1)
    )
    return model 


def plot(losses, val_losses, name):
    f = plt.figure()
    plt.plot(losses)
    plt.plot(val_losses)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(name)


def train(batch_iterator, model, loss_fn, optimiser, n_trainbatch):
    i = 0
    while i < n_trainbatch:
        print("batch index %i" % i)
        batch = next(batch_iterator)
        X = batch[0].to(device)
        Y = batch[1].to(device)
        model.train()
        # forward pass
        y_pred = model(X.float())
        # compute and print loss
        loss = loss_fn(y_pred, Y)
        # Zero the gradients before running the backward pass.
        optimiser.zero_grad()
        # backward pass
        loss.backward()
        # update weights with gradient descent
        optimiser.step()
        i += 1
    return loss.item()


def validate(batch_iterator, model, loss_fn, n_valbatch):
    with torch.no_grad():
        i = 0
        while i < n_valbatch:
            print("validation batch index %i" % i)
            batch = next(batch_iterator)
            X = batch[0].to(device)
            Y = batch[1].to(device)
            model.eval()
            # forward pass
            y_pred = model(X.float())
            # compute and print loss
            val_loss = loss_fn(y_pred, Y)
            i += 1
    return val_loss.item()


def evaluate(batch_iterator, model, n_testbatch):
    preds = []
    groundtruth = []
    with torch.no_grad():
        i = 0
        while i < n_testbatch:
            print("test batch index %i of %i" % (i, n_testbatch))
            batch = next(batch_iterator)
            X = batch[0].to(device)
            Y = batch[1].to(device)
            model.eval()
            # make prediction
            y_pred = model(X.float())
            # store predictions and ground truth labels
            preds.append(y_pred.cpu().numpy())
            groundtruth.append(Y.cpu().numpy())
            i += 1
    return preds, groundtruth

def update_modelpath(modelpath, n_epochs):
    parts = modelpath.split(".")
    new_modelpath = parts[0] + "_" + str(n_epochs) + ".pt"
    return new_modelpath

def main(modelpath, modeltype, n_epochs):
    # if path points to existing model, load it
    if os.path.exists(modelpath):
        print("loading saved model...")
        model = torch.load(modelpath)
    elif modeltype == 1:
        print("new sve model")
        model = model1()
    elif modeltype == 2:
        print("new one hot model")
        model = onehot_model()
    elif modeltype == 3:
        print("new embedding model")
        model = embed_model()
    elif modeltype == 4:
        print("new effect embedding model")
        model = embed_model()
    else:
        print("Usage: python modeltrain.py <modelpath> <modeltype> <n_epochs>")
    model = model.to(device)
    print(model)
    trainparams = {'batch_size': None,
                   'num_workers': 11}
    valparams = {'batch_size': None,
                 'num_workers': 4}
    tstparams = {'batch_size': None,
                 'num_workers': 6}
    n_trainbatch = 33
    n_valbatch = 8
    n_testbatch = 18
    loss_fn = torch.nn.MSELoss(reduction='mean')
    learning_rate = 1e-6
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    beta_mask = np.load('beta_mask.npy')

    losses = []
    val_losses = []

    # initialise early stopping
    tolerance = 10
    no_improvement = 0
    min_val_loss = np.Inf

    t0 = time()
    for t in range(n_epochs):
        print("\n\n\nEpoch = " + str(t))

        if modeltype == 1:
            train_iterator = iter(torch.utils.data.DataLoader(MyIterableDataset('./train/', True), **trainparams))
            valid_iterator = iter(torch.utils.data.DataLoader(MyIterableDataset('./val/', True), **valparams))
            test_iterator = iter(torch.utils.data.DataLoader(MyIterableDataset('./tst/', True), **tstparams))
        elif modeltype == 2:
            trainparams = {'batch_size': None,
                           'num_workers': 3}
            valparams = {'batch_size': None,
                         'num_workers': 4}
            tstparams = {'batch_size': None,
                         'num_workers': 3}
            train_iterator = iter(torch.utils.data.DataLoader(OneHotIterableDataset('./train/', True), **trainparams))
            valid_iterator = iter(torch.utils.data.DataLoader(OneHotIterableDataset('./val/', True), **valparams))
            test_iterator = iter(torch.utils.data.DataLoader(OneHotIterableDataset('./tst/', True), **tstparams))
        elif modeltype == 3:
            train_iterator = iter(torch.utils.data.DataLoader(BasicEmbeddedDataset('./train/', True, 1), **trainparams))
            valid_iterator = iter(torch.utils.data.DataLoader(BasicEmbeddedDataset('./val/', True, 1), **valparams))
            test_iterator = iter(torch.utils.data.DataLoader(BasicEmbeddedDataset('./tst/', True, 1), **tstparams))
        elif modeltype == 4:
            train_iterator = iter(torch.utils.data.DataLoader(EffectEmbeddingDataset('./train/', True, 2, beta_mask), **trainparams))
            valid_iterator = iter(torch.utils.data.DataLoader(EffectEmbeddingDataset('./val/', True, 2, beta_mask), **valparams))
            test_iterator = iter(torch.utils.data.DataLoader(EffectEmbeddingDataset('./tst/', True, 2, beta_mask), **tstparams))

        print("training...")
        # full training step
        loss = train(train_iterator, model, loss_fn, optimiser, n_trainbatch)
        losses.append(loss)
        print(loss)
        # validation step
        print("validating...")
        val_loss = validate(valid_iterator, model, loss_fn, n_valbatch)
        val_losses.append(val_loss)
        print(val_loss)
        # early stopping
        # check conditions for early stopping
        if val_loss < min_val_loss:
            no_improvement = 0
            min_val_loss = val_loss
        else:
            no_improvement += 1
        if t > 5 and no_improvement == tolerance:
            print("min validation loss: %f"%min_val_loss)
            print("no improvement for %i epochs"%no_improvement)
            print("STOPPING EARLY")
            break
    t1 = time()
    print("time taken: %f s" % (t1 - t0))
    modelpath = update_modelpath(modelpath, t)
    torch.save(model,modelpath)
    # plot
    plot(losses, val_losses, modelpath.split(".")[0]+".png")
    # evaluate
    preds, groundtruths = evaluate(test_iterator,model,n_testbatch)
    pr = np.concatenate(preds, 1).ravel()
    gt = np.concatenate(groundtruths, 1).ravel()
    r = stats.pearsonr(pr, gt)
    print(r)


if __name__ == "__main__":
    main(modelpath=sys.argv[1], modeltype = int(sys.argv[2]), n_epochs=int(sys.argv[3]))
