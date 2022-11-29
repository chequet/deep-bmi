### architecture gridsearch with ray tune
import sys

import torch
import torch.optim as optim
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from MyIterableDataset3 import *
from OneHotIterableDataset import *
from BasicEmbeddedDataset import *
import os
from modeltrain import train_val_split
from FlexibleNet import *
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# PARAMS TO CHANGE ============================
N_SNPS = 100
N_INPUTS = 300
N_EPOCHS = 10
ENCODING = 2
BATCH_SIZE = 4096
#==============================================

# def cross_validation(data, k=5):
    # leave for now

def get_dataloaders(data_directory, type, trainworkers=4, valworkers=2, n_train=48):
    train_files, val_files = train_val_split(data_directory + 'train/',n_train=n_train)
    n_train = len(train_files)
    n_val = len(val_files)
    trainparams = {'batch_size': None,
                   'num_workers': trainworkers}
    valparams = {'batch_size': None,
                 'num_workers': valworkers}
    train_iterator = None
    valid_iterator = None
    if type == 1:
        train_iterator = iter(
            torch.utils.data.DataLoader(MyIterableDataset(data_directory + 'train/', train_files, True),
                                        **trainparams))
        valid_iterator = iter(
            torch.utils.data.DataLoader(MyIterableDataset(data_directory + 'train/', val_files, True),
                                        **valparams))
    elif type == 2:
        train_iterator = iter(
            torch.utils.data.DataLoader(OneHotIterableDataset(data_directory + 'train/', train_files, True),
                                        **trainparams))
        valid_iterator = iter(
            torch.utils.data.DataLoader(OneHotIterableDataset(data_directory + 'train/', val_files, True),
                                        **valparams))
    elif type == 3:
        train_iterator = iter(
            torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory + 'train/', train_files, True, 1),
                                        **trainparams))
        valid_iterator = iter(
            torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory + 'train/', val_files, True, 1),
                                        **valparams))
    elif type == 4:
        train_iterator = iter(
            torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory + 'train/', train_files, True, 2),
                                        **trainparams))
        valid_iterator = iter(
            torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory + 'train/', val_files, True, 2),
                                        **valparams))
    return train_iterator, valid_iterator, n_train, n_val

def train(config, checkpoint_dir=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    # set up model according to config
    model = FlexibleNet(config["arch"],config["dropout"],config["activation"])
    model.to(device)
    # choose loss fn based on config
    if config["loss"] == "MSE":
        loss_fn = nn.MSELoss(reduction='mean')
    elif config["loss"] == "huber":
        loss_fn = nn.HuberLoss()
    learning_rate = config['lr']
    # choose optimiser based on config
    if config["optim"] == 'adam':
        optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    elif config["optim"] == "sgd":
        optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif config["optim"] == "rmsprop":
        optimiser = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif config["optim"] == "adamw":
        optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif config["optim"] == "spadam":
        optimiser = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
    elif config["optim"] == "adamax":
        optimiser = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    elif config["optim"] == "nadam":
        optimiser = torch.optim.NAdam(model.parameters(), lr=learning_rate)
    elif config["optim"] == "radam":
        optimiser = torch.optim.RAdam(model.parameters(), lr=learning_rate)
    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimiser.load_state_dict(optimizer_state)
    data_directory = "/data/" + str(N_SNPS) + "_data/"
    # train
    for epoch in range(N_EPOCHS):
        train_iterator, valid_iterator, n_train, n_val = get_dataloaders(data_directory, type=ENCODING)
        # TRAIN
        i = 0
        while i < n_train:
            batch = next(train_iterator)
            # print("batch index %i" % i, end='\r')
            X = batch[0].to(device)
            Y = batch[1].to(device)
            # Zero the gradients
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
            i+=1
        # VALIDATE
        with torch.no_grad():
            val_loss = 0.0
            val_r2 = 0.0
            val_r = 0.0
            i = 0
            while i < n_val:
                # print("validation batch index %i" % i, end='\r')
                batch = next(valid_iterator)
                X = batch[0].to(device)
                Y = batch[1].to(device)
                model.eval()
                # forward pass
                y_pred = model(X.float())
                # compute and print loss
                loss = loss_fn(y_pred, Y)
                val_loss += loss.cpu().numpy()
                val_r2 += r2_score(y_pred.cpu().numpy(), Y.cpu().numpy())
                val_r += pearsonr(y_pred.ravel().cpu().numpy(), Y.ravel().cpu().numpy())[0]
                i += 1
        # Save a Ray Tune checkpoint & report score to Tune
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (model.state_dict(), optimiser.state_dict()), path)
        tune.report(r2=(val_r2 / i), loss=(val_loss / i), r=(val_r / i))

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

def main():
    # generate architectures
    layer_params = [
        [2, 2, 2],
        [1, 10],
        [10, 10],
        [2, 1, 2, 1]
    ]
    architectures = []
    for r in layer_params:
        a = make_architecture(N_INPUTS, 1, r)
        architectures.append(a)
    print("\nARCHITECTURE CHOICES")
    print(architectures)
    # define config
    config = {
        "arch": tune.grid_search(architectures),
        "activation": tune.grid_search(["ELU", "ReLU","LeakyReLU"]),
        "dropout": tune.grid_search([0,0.1,0.2,0.3]),
        "optim": tune.choice(["adam","adamw","adamax","radam"]), #"nadam","spadam","sgd","rmsprop",
        "lr": tune.loguniform(1e-4, 1e-1),
        "loss": tune.grid_search(["huber"])
    }
    scheduler = ASHAScheduler(
        max_t=N_EPOCHS,
        grace_period=1,
        reduction_factor=2)
    # run
    print("running...")
    result = tune.run(
        tune.with_parameters(train),
        resources_per_trial={"cpu": 60, "gpu": 1},
        config=config,
        metric="loss",
        mode="min",
        num_samples=1,
        scheduler=scheduler,
        max_concurrent_trials=3
    )
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation r2: {}".format(
        best_trial.last_result["r2"]))
    df = result.results_df
    sorted = df.sort_values('loss')
    #TODO filter for NaN before printing
    print("\n\n====================================================================\n")
    print(sorted)
    filename = "grid_search2/encoding" + str(ENCODING) + "_" + str(N_SNPS) + "_tuneresults.csv"
    sorted.to_csv(filename)

if __name__ == "__main__":
    main()






