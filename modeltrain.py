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
from Net4 import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

# print("\n\nSMALL BATCH DEMO\n\n")

def sve_model(n_inputs, reduction_factor, dropout):
    n_features = n_inputs
    n_out1 = math.ceil(n_features/reduction_factor)
    n_out2 = math.ceil(n_out1/reduction_factor)
    n_out3 = math.ceil(n_out2/reduction_factor)
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, n_out1),
        torch.nn.Dropout(dropout),
        torch.nn.Sigmoid(),
        torch.nn.Linear(n_out1, n_out2),
        torch.nn.Dropout(dropout),
        torch.nn.Sigmoid(),
        torch.nn.Linear(n_out2, n_out3),
        torch.nn.Dropout(dropout),
        torch.nn.Sigmoid(),
        torch.nn.Linear(n_out3, 1)
    )
    return model

def onehot_model(n_inputs, reduction_factor, dropout):
    n_features = n_inputs*3
    n_out1 = math.ceil(n_features/reduction_factor)
    n_out2 = math.ceil(n_out1/reduction_factor)
    n_out3 = math.ceil(n_out2/reduction_factor)
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, n_out1),
        torch.nn.Dropout(dropout),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(n_out1, n_out2),
        torch.nn.Dropout(dropout),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(n_out2, n_out3),
        torch.nn.Dropout(dropout),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(n_out3, 1)
    )
    return model

def embed_model(n_inputs, reduction_factor, dropout):
    n_features = n_inputs*2
    n_out1 = math.ceil(n_features/reduction_factor)#(reduction_factor**2))
    n_out2 = math.ceil(n_out1/reduction_factor)
    n_out3 = math.ceil(n_out2/reduction_factor)

    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, n_out1),
        torch.nn.Dropout(dropout),
        torch.nn.Sigmoid(),
        torch.nn.Linear(n_out1, n_out2),
        torch.nn.Dropout(dropout),
        torch.nn.Sigmoid(),
        torch.nn.Linear(n_out2, n_out3),
        torch.nn.Dropout(dropout),
        torch.nn.Sigmoid(),
        torch.nn.Linear(n_out3, 1)
    )
    return model

def embed_model2(n_inputs, reduction_factor, dropout):
    n_features = n_inputs*2
    n_out1 = math.ceil(n_features/reduction_factor)#(reduction_factor**2))
    n_out2 = math.ceil(n_out1/reduction_factor)
    n_out3 = math.ceil(n_out2/reduction_factor)
    n_out4 = math.ceil(n_out3/reduction_factor)
    n_out5 = math.ceil(n_out4/reduction_factor)

    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, n_out1),
        torch.nn.Dropout(dropout),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(n_out1, n_out2),
        torch.nn.Dropout(dropout),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(n_out2, n_out3),
        torch.nn.Dropout(dropout),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(n_out3, n_out4),
        torch.nn.Dropout(dropout),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(n_out4, n_out5),
        torch.nn.Dropout(dropout),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(n_out5, 1),
        )
    return model

def embed_model_clf(n_inputs, reduction_factor, dropout):
    n_features = n_inputs*2
    n_features = n_inputs*2
    n_out1 = math.ceil(n_features/reduction_factor)#(reduction_factor**2))
    n_out2 = math.ceil(n_out1/reduction_factor)
    n_out3 = math.ceil(n_out2/reduction_factor)
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, n_out1),
        torch.nn.Dropout(dropout),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(n_out1, n_out2),
        torch.nn.Dropout(dropout),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(n_out2, n_out3),
        torch.nn.Dropout(dropout),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(n_out3, 2)
    )
    return model

def conv_model_clf():
    n_features = 100000
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, 500),
        torch.nn.Dropout(0.3),
        torch.nn.LeakyReLU(),
        torch.nn.Conv1d(in_channels=500, out_channels=200, kernel_size=(3,), stride=(3,), padding=(1,)),
        torch.nn.BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True),
        torch.nn.Dropout(0.3),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(200, 10),
        torch.nn.Flatten(),
        torch.nn.Dropout(0.3),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(10, 2)
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

def plot_accs(train_accs, val_accs, name):
    f = plt.figure()
    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(name)

def binary_acc(y_pred, y):
    correct_results_sum = len(np.where(y_pred == y)[0])
    acc = float(correct_results_sum) / len(y)
    acc = np.round(acc * 100)

    return acc


def train(batch_iterator, model, loss_fn, optimiser, n_trainbatch, clf):
    i = 0
    acc = 0
    while i < n_trainbatch:
        print("batch index %i" % i)
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
    return loss.item(), acc


def validate(batch_iterator, model, loss_fn, n_valbatch, clf):
    with torch.no_grad():
        i = 0
        # null accuracy for regression
        acc = 0
        while i < n_valbatch:
            print("validation batch index %i" % i)
            batch = next(batch_iterator)
            X = batch[0].to(device)
            Y = batch[1].to(device)
            model.eval()
            # forward pass
            y_pred = model(X.float())
            #print("y pred:")
            #print(y_pred)
            if clf:
                # X = X.reshape(X.shape[0],X.shape[1],1)
                # print(X.shape)
                y_round = y_pred.argmax(dim=1)
                acc = binary_acc(y_round.cpu(), Y.flatten().cpu())
                Y = Y.flatten().long()
            # compute and print loss
            val_loss = loss_fn(y_pred, Y)
            i += 1
    return val_loss.item(), acc

def evaluate(batch_iterator, model, loss_fn, n_testbatch, clf):
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
            # round to nearest integer if doing classification instead of regression
            if clf:
                # X = X.reshape(X.shape[0],X.shape[1],1)
                # print(X.shape)
                y_pred = y_pred.argmax(dim=1)
            # store predictions and ground truth labels
            preds.append(y_pred.cpu().numpy())
            groundtruth.append(Y.cpu().numpy())
            test_loss = loss_fn(y_pred, Y)
            i += 1
    return preds, groundtruth, test_loss

def update_modelpath(modelpath, n_epochs):
    parts = modelpath.split(".")
    new_modelpath = parts[0] + "_" + str(n_epochs) + ".pt"
    return new_modelpath

def train_val_split(train_dir, n_train=99):
    # for now let's just hard code this to do a nice round number of training samples
    # that divides easily among workers
    files = os.listdir(train_dir)
    np.random.shuffle(files)
    #n_train = int(np.ceil(len(files)*proportion))
    trainfiles = files[:n_train]
    valfiles = files[n_train:]
    return trainfiles, valfiles

def main(modelpath, modeltype, n_epochs, n_inputs):

    REDUCTION_FACTOR = 10
    DROPOUT = 0.1

    # if path points to existing model, load it
    if os.path.exists(modelpath):
        print("loading saved model...")
        model = torch.load(modelpath)
    elif modeltype == 1:
        print("new sve modeln n inputs: %i"%n_inputs)
        model = sve_model(n_inputs,REDUCTION_FACTOR,DROPOUT)
    elif modeltype == 2:
        print("new one hot model")
        model = onehot_model(n_inputs,REDUCTION_FACTOR,DROPOUT)
    elif modeltype == 3:
        print("new embedding model, n inputs: %i"%n_inputs)
        model = embed_model(n_inputs,REDUCTION_FACTOR,DROPOUT)
    elif modeltype == 4:
        print("new effect embedding model")
        model = embed_model(n_inputs,REDUCTION_FACTOR,DROPOUT)
    elif modeltype == 5:
        print("new embedding classification model")
        model = embed_model_clf(n_inputs,REDUCTION_FACTOR,DROPOUT)
    elif modeltype == 6:
        print("new CNN classification model")
        model = conv_model_clf()   # define the network
    elif modeltype == 7:
        print("new deeper embedding model")
        model = embed_model2(n_inputs,REDUCTION_FACTOR,DROPOUT)   # define the network
    else:
        print("Usage: python modeltrain.py <modelpath> <modeltype> <n_epochs> <n_inputs>")
    model = model.to(device)
    print(model)

    # initialise training and validation sets
    data_directory = "../new_data_" + str(n_inputs)
    train_files, val_files = train_val_split(data_directory + '/train/')
    test_files = os.listdir(data_directory + '/tst/')

    trainparams = {'batch_size': None,
                   'num_workers': 11}
    valparams = {'batch_size': None,
                 'num_workers': 4}
    tstparams = {'batch_size': None,
                 'num_workers': 6}
    n_trainbatch = len(train_files)
    n_valbatch = len(val_files)
    n_testbatch = len(test_files)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    learning_rate = 1e-2
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.9)
    #beta_mask = np.load('beta_mask.npy')
    clf = False

    losses = []
    val_losses = []
    accs = []
    val_accs = []


    # initialise early stopping
    tolerance = 50
    no_improvement = 0
    min_val_loss = np.Inf

    t0 = time()
    for t in range(n_epochs):
        print("\n\n\nEpoch = " + str(t))

        if modeltype == 1:
            train_iterator = iter(torch.utils.data.DataLoader(MyIterableDataset(data_directory+'/train/', True), **trainparams))
            valid_iterator = iter(torch.utils.data.DataLoader(MyIterableDataset(data_directory+'/val/', True), **valparams))
            test_iterator = iter(torch.utils.data.DataLoader(MyIterableDataset(data_directory+'/tst/', True), **tstparams))
        elif modeltype == 2:
            trainparams = {'batch_size': None,
                           'num_workers': 3}
            valparams = {'batch_size': None,
                         'num_workers': 4}
            tstparams = {'batch_size': None,
                         'num_workers': 3}
            train_iterator = iter(torch.utils.data.DataLoader(OneHotIterableDataset(data_directory+'/train/', True), **trainparams))
            valid_iterator = iter(torch.utils.data.DataLoader(OneHotIterableDataset(data_directory+'/val/', True), **valparams))
            test_iterator = iter(torch.utils.data.DataLoader(OneHotIterableDataset(data_directory+'/tst/', True), **tstparams))
        elif modeltype == 3 or modeltype == 7:
            train_iterator = iter(torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory+'/train/', train_files, True, 1), **trainparams))
            valid_iterator = iter(torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory+'/train/', val_files, True, 1), **valparams))
            test_iterator = iter(torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory+'/tst/', test_files, True, 1), **tstparams))
        elif modeltype == 4:
            train_iterator = iter(torch.utils.data.DataLoader(EffectEmbeddingDataset(data_directory+'/train/', True, 2, beta_mask), **trainparams))
            valid_iterator = iter(torch.utils.data.DataLoader(EffectEmbeddingDataset(data_directory+'/val/', True, 2, beta_mask), **valparams))
            test_iterator = iter(torch.utils.data.DataLoader(EffectEmbeddingDataset(data_directory+'/tst/', True, 2, beta_mask), **tstparams))
        elif modeltype == 5 or modeltype == 6:
            train_iterator = iter(torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory+'/train/', True, 1, clf=True), **trainparams))
            valid_iterator = iter(torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory+'/val/', True, 1, clf=True), **valparams))
            test_iterator = iter(torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory+'/tst/', True, 1, clf=True), **tstparams))
            loss_fn = torch.nn.CrossEntropyLoss()
            clf = True
        
        print("training...")
        # full training step
        train_loss, train_acc = train(train_iterator, model, loss_fn, optimiser, n_trainbatch, clf)
        losses.append(float(train_loss))
        print(train_loss)
        del(train_loss)
        # update LR
        #scheduler.step()
        state = optimiser.state_dict()['param_groups'][0]
        lr = state['lr']
        #print("lr: %f" %lr)
        # validation step
        print("validating...")
        val_loss, val_acc = validate(valid_iterator, model, loss_fn, n_valbatch, clf)
        val_losses.append(float(val_loss))
        print(val_loss)
        del(val_loss)
        if clf:
            accs.append(int(train_acc))
            print("training accuracy: %f"%train_acc)
            del(train_acc)
            val_accs.append(int(val_acc))
            print("validation accuracy: %f"%val_acc)
            del(val_acc)
        # early stopping
        # check conditions for early stopping
        # if val_loss < min_val_loss:
        #     no_improvement = 0
        #     min_val_loss = val_loss
        # else:
        #     no_improvement += 1
        # if t > 5 and no_improvement == tolerance:
        #     print("min validation loss: %f"%min_val_loss)
        #     print("no improvement for %i epochs"%no_improvement)
        #     print("STOPPING EARLY")
        #     break
    t1 = time()
    print("time taken: %f s" % (t1 - t0))
    modelpath = update_modelpath(modelpath, t)
    torch.save(model,modelpath)
    # plot
    plot(losses, val_losses, modelpath.split(".")[0]+".png")
    if clf:
        plot_accs(accs, val_accs, modelpath.split(".")[0]+"_accuracy.png")
    # evaluate
    preds, groundtruths, test_loss = evaluate(test_iterator, model, loss_fn, n_testbatch, clf)
    pr = np.concatenate(preds).ravel()
    gt = np.concatenate(groundtruths).ravel()

    if clf:
        acc = binary_acc(pr,gt)
        resultstring = "test accuracy: " + str(acc)
    else:
        r = stats.pearsonr(pr, gt)
        resultstring = "pearson r coeff: " + str(r[0])
    resultstring += (" test loss: "+str(test_loss.item()))
    print(resultstring)
    print("plot path: %s"%(modelpath.split(".")[0]+".png"))
    # save
    resultfile = modelpath.split(".")[0]+".txt"
    with open(resultfile, 'w') as f:
        f.write(resultstring)

if __name__ == "__main__":
    main(modelpath=sys.argv[1], modeltype = int(sys.argv[2]), n_epochs=int(sys.argv[3]), n_inputs=int(sys.argv[4]))
