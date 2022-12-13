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
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# PARAMS TO CHANGE ============================
N_SNPS = int(sys.argv[1])
N_INPUTS = int(sys.argv[2])
N_EPOCHS = int(sys.argv[4])
ENCODING = int(sys.argv[3])
BATCH_SIZE = 4096
REDUCTIONS = [50,10,10]
PATH = 'BNN_' + str(N_SNPS) + '_huber_radam_leakyrelu_dropout05_' + str(ENCODING)
#==============================================

def train_and_validate_BNN(arch, data_directory, train_set, val_set):
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
    best_val_loss = np.inf
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
        print("training loss: %f" % loss)
        # log training loss w tensorboard
        writer.add_scalar("Loss/train", loss, t)
        # validate on every 10th epoch only
        if t % 10 == 0:
            loss, r, r2, ci_acc, under_ci_upper, over_ci_lower = evaluate_regression(model, valid_iterator, val_set, loss_fn)
            print("validation loss: %f" % loss)
            print("CI acc: {:.2f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}".format(ci_acc, under_ci_upper,
                                                                                      over_ci_lower))
            print("pearson r: %f" % r)
            writer.add_scalar("Loss/val", loss, t)
            writer.add_scalar("ci_acc", ci_acc, t)
            writer.add_scalar("Pearson_R", r, t)
            writer.add_scalar("R2", r2, t)
            # check conditions for early stopping
            if t % 10 == 0:
                print("no improvement for %i epochs" % no_improvement)
            if loss < best_val_loss:
                no_improvement = 0
                best_val_loss = loss
            else:
                no_improvement += 1
            # 30 epoch grace period
            if t > 30 and no_improvement >= tolerance:
                print("best validation loss: %f" % best_val_loss)
                print("STOPPING EARLY\n\n")
                break
    writer.flush()
    writer.close()
    torch.save(model, PATH)
    return loss, ci_acc, under_ci_upper, over_ci_lower, r, r2, t


def evaluate_regression(model, valid_iterator, samples, loss_fn, std_multiplier = 2):
    preds = []
    gt = []
    i = 0
    while i < len(samples):
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
    r2 = r2_score(preds, gt)
    r = pearsonr(preds, gt)[0]
    return loss, r, r2, ic_acc, (ci_upper >= gt).mean(), (ci_lower <= gt).mean()



def main(modelpath, n_epochs):


if __name__ == "__main__":
    main(modelpath=sys.argv[1], n_epochs=int(sys.argv[2]))