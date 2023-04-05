import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from BayesianNN import *
from OneHotIterableDataset import *
from modeltrain2 import k_fold_split, get_train_files, get_dataloader, make_architecture
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import csv
import sys

# PARAMS TO CHANGE ============================
N_SNPS = int(sys.argv[1])
N_INPUTS = int(sys.argv[2])
ENCODING = int(sys.argv[3])
N_EPOCHS = int(sys.argv[4])
BATCH_SIZE = 4096
REDUCTIONS = [100,5,2,5,2]
PATH = 'BNN_' + str(N_SNPS) + '_mse_adamw_elu_dropout0_' + str(ENCODING) + ".pt"
#==============================================

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

def train_and_validate_BNN(arch, data_directory, train_set, val_set):
    # new model
    # -------------PARAMS-----------------------------------------------
    model = BayesianNN(arch, 0, 'ELU').to(device)
    learning_rate = 0.001
    #loss_fn = torch.nn.HuberLoss()
    loss_fn = torch.nn.MSELoss(reduction='mean')
    #loss_fn = torch.nn.L1Loss()
    optimiser = optim.Adamw(model.parameters(), lr=0.01)
    # ------------------------------------------------------------------
    print(model)
    # # initialise summary writer for tensorboard
    writer = SummaryWriter()
    # initialise early stopping
    tolerance = 3
    no_improvement = 0
    best_ci_acc = 0
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
            loss = model.sample_elbo(inputs=X.float(),
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
            val_loss, r, r2, ci_acc, under_ci_upper, over_ci_lower = \
                evaluate_regression(model, valid_iterator, val_set, loss_fn)
            print("validation loss: %f" % val_loss)
            print("CI acc: {:.3f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}".format(ci_acc, under_ci_upper,
                                                                                      over_ci_lower))
            print("pearson r: %f" % r)
            writer.add_scalar("Loss/val", val_loss, t)
            writer.add_scalar("ci_acc", ci_acc, t)
            writer.add_scalar("Pearson_R", r, t)
            writer.add_scalar("R2", r2, t)
            # check conditions for early stopping - use CI acc as criteria
            if t % 10 == 0:
                print("no improvement for %i epochs" % (no_improvement*10))
            if ci_acc > best_ci_acc:
                no_improvement = 0
                best_ci_acc = ci_acc
            else:
                no_improvement += 1
            # 30 epoch grace period
            if t > 30 and no_improvement >= tolerance:
                print("best c.i. accuracy: %f" % best_ci_acc)
                print("STOPPING EARLY\n\n")
                break
    writer.flush()
    writer.close()
    torch.save(model, PATH)
    return val_loss, ci_acc, under_ci_upper, over_ci_lower, r, r2, t


def evaluate_regression(model, valid_iterator, val_set, loss_fn, n_samples=100,  std_multiplier=1):
    # preds = []
    # get ground truth to compare to
    gt = []
    ins = []
    i = 0
    while i < len(val_set):
        batch = next(valid_iterator)
        X = batch[0].to(device)
        y = batch[1].to(device)
        # y_pred = model(X.float())
        # preds.append(y_pred.detach().cpu().numpy())
        ins.append(X)
        gt.append(y)
        i += 1
    # preds = np.concatenate(preds).ravel()
    gt = torch.cat(gt)
    ins = torch.cat(ins)
    # calculate loss just for last batch for early stopping purposes
    model.eval()
    loss = model.sample_elbo(inputs=X.float(),
                                     labels=y,
                                     criterion=loss_fn,
                                     sample_nbr=3)
    # get a bunch of predictions to compare!
    print("generating %i predictions..."%n_samples)
    preds = [model(ins.float()) for i in range(n_samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    # print(means[:10])
    # print(gt[:10])
    # print(stds[:10])
    ci_upper = means + (std_multiplier * stds)
    # print(ci_upper[:10])
    ci_lower = means - (std_multiplier * stds)
    # print(ci_lower[:10])
    ci_acc = (ci_lower <= gt) * (ci_upper >= gt)
    # print(ci_acc[:10])
    ci_acc = ci_acc.float().mean()
    r2 = r2_score(means.detach().numpy(), gt.detach().numpy())
    r = pearsonr(means.detach().numpy().ravel(), gt.detach().numpy().ravel())[0]
    return loss.detach().numpy(), r, r2, ci_acc, (ci_upper >= gt).float().mean(), (ci_lower <= gt).float().mean()



def main():
    arch = make_architecture(N_INPUTS, 1, REDUCTIONS)
    # save results for printing and persisting
    results = {
        'validation_sets': [], 'validation_loss': [],
        'ci_acc': [], 'under_ci_upper': [], 'over_ci_lower': [],
        'validation_r': [], 'validation_r2': [], 'n_epochs': []
               }
    # 5-fold cross validation
    data_directory = "/data/" + str(N_SNPS) + "_data_relabelled/"
    print(data_directory)
    cross_val_partitions = k_fold_split(data_directory + 'train/')

    for val_set in cross_val_partitions:
        results['validation_sets'].append(val_set)
        train_set = get_train_files(data_directory + 'train/', val_set)
        print("\nvalidation set:")
        print(val_set)
        val_loss, ci_acc, under_upper, over_lower, val_r, val_r2, t = \
            train_and_validate_BNN(arch, data_directory, train_set, val_set)
        results['validation_loss'].append(val_loss)
        results['ci_acc'].append(ci_acc)
        results['over_ci_lower'].append(over_lower)
        results['under_ci_upper'].append(under_upper)
        results['validation_r'].append(val_r)
        results['validation_r2'].append(val_r2)
        results['n_epochs'].append(t)
    # save results
    results_path = '../results/' + PATH + '_results.csv'
    with open(results_path, 'w') as f:
        w = csv.writer(f)
        w.writerows(results.items())
    # print interesting results
    print('mean validation loss: %f' % np.mean(np.array(results['validation_loss'])))
    print('best validation loss: %f' % np.min(np.array(results['validation_loss'])))
    print('mean validation ci acc: %f' % np.mean(np.array(results['ci_acc'])))
    print('best validation ci acc: %f' % np.max(results['ci_acc']))
    print('mean validation r: %f' % np.mean(np.array(results['validation_r'])))
    print('best validation r: %f' % np.max(np.array(results['validation_r'])))
    print('mean validation r2: %f' % np.mean(np.array(results['validation_r2'])))
    print('best validation r2: %f' % np.max(np.array(results['validation_r2'])))
    print('mean number of epochs: %i' % np.mean(np.array(results['n_epochs'])))
    print('epoch range: %i' % (np.max(np.array(results['n_epochs'])) - np.min(np.array(results['n_epochs']))))


if __name__ == "__main__":
    main()