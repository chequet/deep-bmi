import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from time import time
from BasicEmbeddedDataset import *
from torch.utils.tensorboard import SummaryWriter
from modeltrain import *
from BayesianNN import *

## CURRENTLY FIXED AT 10K INPUTS

def train_BNN(batch_iterator, model, loss_fn, optimiser, n_trainbatch, clf):
    i = 0
    acc = 0
    while i < n_trainbatch:
        print("batch index %i" % i, end='\r')
        # Zero the gradients before running the backward pass.
        optimiser.zero_grad()
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
        # compute and print loss
        loss = model.sample_elbo(inputs=X,
                                     labels=Y,
                                     criterion=loss_fn,
                                     sample_nbr=3)
                                     # complexity_cost_weight=1 / X.shape[0])

        # backward pass
        loss.backward()
        # update weights with gradient descent
        optimiser.step()
        i += 1
    return loss.item(), acc

def evaluate_regression(model, valid_iterator, samples,loss_fn, std_multiplier = 2):
    preds = []
    gt = []
    i = 0
    while i < samples:
        batch = next(valid_iterator)
        X = batch[0].to(device)
        y = batch[1].to(device)
        model.eval()
        y_pred = model(X.float())
        preds.append(y_pred.detach().cpu().numpy())
        gt.append(y.detach().cpu().numpy())
        i += 1
    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    # calculate loss just for last batch for early stopping purposes
    loss = model.sample_elbo(inputs=X.float(),
                                     labels=y,
                                     criterion=loss_fn,
                                     sample_nbr=3)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= gt) * (ci_upper >= gt)
    ic_acc = ic_acc.mean()
    return loss, ic_acc, (ci_upper >= gt).mean(), (ci_lower <= gt).mean()

def main(modelpath, n_epochs):
    REDUCTION_FACTOR = 2
    DROPOUT = 0.2
    ACTIVATION = 'ELU'
    LAYERS = [19988, 2000, 1000, 500, 250, 125, 60, 30, 15, 1]
    model = BayesianNN(LAYERS, DROPOUT, ACTIVATION)
    model = model.to(device)
    print(model)

    # initialise training and validation sets
    n_inputs = 9994
    data_directory = "../old_data/" + str(n_inputs) + "_data/"
    train_files, val_files = train_val_split(data_directory + '/train/')
    test_files = os.listdir(data_directory + '/tst/')

    trainparams = {'batch_size': None,
                   'num_workers': 11}
    valparams = {'batch_size': None,
                 'num_workers': 4}
    n_trainbatch = len(train_files)
    print("n train: "+ str(n_trainbatch))
    n_valbatch = len(val_files)
    print("n val: "+ str(n_valbatch))
    loss_fn = torch.nn.MSELoss(reduction='mean')
    learning_rate = 1e-4
    optimiser = optim.Adamax(model.parameters(), lr=learning_rate)

    # initialise early stopping with ci_acc
    tolerance = 3
    no_improvement = 0
    best_ci_acc = 0

    losses = []
    val_losses = []

    t0 = time()
    for t in range(n_epochs):
        print("\n\n\nEpoch = " + str(t))
        train_iterator = iter(
            torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory + '/train/', train_files, True, 1),
                                        **trainparams))
        valid_iterator = iter(
            torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory + '/train/', val_files, True, 1),
                                        **valparams))
        print("training...")
        # full training step
        train_loss, train_acc = train(train_iterator, model, loss_fn, optimiser, n_trainbatch, False)
        # add extra loss step!

        losses.append(float(train_loss))
        print("loss: {:.2f}".format(train_loss))
        # log training loss w tensorboard
        writer.add_scalar("Loss/train", train_loss, t)
        # print("lr: %f" %lr)
        if t%10==0:
            # validation step - only do every ten epochs to save computational complexity
            print("validating...")
            val_loss, ci_acc, under_ci_upper, over_ci_lower = evaluate_regression(model,
                                                                        valid_iterator,
                                                                        samples=n_valbatch,
                                                                        loss_fn=loss_fn,
                                                                        std_multiplier=3)
            print("validation loss: {:.2f}".format(val_loss))
            print("CI acc: {:.2f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}".format(ci_acc, under_ci_upper,
                                                                                      over_ci_lower))
            writer.add_scalar("ci_acc", ci_acc, t)
            writer.add_scalar("Loss/val", val_loss, t)
            # early stopping
            # check conditions for early stopping
            if ci_acc > best_ci_acc:
                no_improvement = 0
                best_ci_acc = ci_acc
            else:
                no_improvement += 1
            if t > 5 and no_improvement == tolerance:
                print("best confidence interval accuracy: %f" % best_ci_acc)
                print("STOPPING EARLY")
                break
    t1 = time()
    print("time taken: %f s" % (t1 - t0))
    modelpath = update_modelpath(modelpath, t)
    torch.save(model, modelpath)
    # plot
    plot(losses, val_losses, modelpath.split(".")[0] + ".png")
    print("plot path: %s"%(modelpath.split(".")[0]+".png"))


if __name__ == "__main__":
    main(modelpath=sys.argv[1], n_epochs=int(sys.argv[2]))