import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from time import time
from BasicEmbeddedDataset import *
from torch.utils.tensorboard import SummaryWriter
from modeltrain import *
from BayesianNN import *
from MyIterableDataset3 import *
from OneHotIterableDataset import *
from modeltrain2 import k_fold_split, make_architecture, get_train_files, get_dataloader

# PARAMS TO CHANGE ============================
N_SNPS = int(sys.argv[1])
N_INPUTS = int(sys.argv[2])
N_EPOCHS = int(sys.argv[4])
ENCODING = int(sys.argv[3])
BATCH_SIZE = 4096
REDUCTIONS = [50,10,10]
PATH = 'BNN_' + str(N_SNPS) + '_huber_radam_leakyrelu_dropout05_' + str(ENCODING)
#==============================================

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

def train_and_validate(arch, data_directory, train_set, val_set):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    # new model
    # -------------PARAMS-----------------------------------------------
    model = BayesianNN(arch, 0.5, 'LeakyReLU').to(device)
    learning_rate = 0.0001
    loss_fn = torch.nn.HuberLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    # ------------------------------------------------------------------
    print(model)
    # # initialise summary writer for tensorboard
    writer = SummaryWriter()
    # initialise early stopping
    tolerance = 10
    no_improvement = 0
    best_val_r = -np.inf
    for t in range(N_EPOCHS):
        print("epoch %i" % t)
        train_iterator = get_dataloader(data_directory, ENCODING, 8, train_set)
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
        if t % 10 == 0:
            print("no improvement for %i epochs" % t)
        if r > best_val_r:
            no_improvement = 0
            best_val_r = r
        else:
            no_improvement += 1
        # 30 epoch grace period
        if t > 30 and no_improvement >= tolerance:
            print("best validation r: %f" % best_val_r)
            print("STOPPING EARLY\n\n")
            break
    writer.flush()
    writer.close()
    torch.save(model, PATH)
    return loss, r, r2, t


def evaluate_regression(model, valid_iterator, samples, loss_fn, std_multiplier = 2):
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

    # initialise training and validation sets

    data_directory = "../old_data/" + str(N_FEATURES)+ "_data/"
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
    tolerance = 6
    no_improvement = 0
    best_ci_acc = 0

    losses = []
    val_losses = []

    t0 = time()
    for t in range(n_epochs):
        print("\n\n\nEpoch = " + str(t))
        train_iterator = iter(
            torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory + '/train/', train_files, True,2),
                                        **trainparams))
        valid_iterator = iter(
            torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory + '/train/', val_files, True,2),
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
            val_losses.append(val_loss)
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