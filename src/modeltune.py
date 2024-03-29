### architecture gridsearch with ray tune
import sys

import torch
import torch.optim as optim
import torch.nn as nn
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from generators.MyIterableDataset3 import *
from generators.OneHotIterableDataset import *
from generators.BasicEmbeddedDataset import *
from generators.EffectEmbeddingDataset import *
import os
from src.modeltrain import train_val_split
from nets.FlexibleNet import *
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import warnings
import pickle

# PARAMS TO CHANGE ============================
N_SNPS = int(sys.argv[1])
N_INPUTS = int(sys.argv[2])
N_EPOCHS = 20
ENCODING = int(sys.argv[3])
BATCH_SIZE = 4096
#==============================================

def get_dataloader(data_directory, encoding, workers, files, beta_mask = None):
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
    elif encoding == 5:
        dataloader = iter(torch.utils.data.DataLoader
                          (EffectEmbeddingDataset(data_directory + 'train/', files, True, 1, beta_mask),**params))
    elif encoding == 6:
        dataloader = iter(torch.utils.data.DataLoader
                          (EffectEmbeddingDataset(data_directory + 'train/', files, True, 2, beta_mask),**params))
    return dataloader

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
    data_directory = "/data/" + str(N_SNPS) + "_data_relabelled/"
    beta_mask = pickle.load(open("/data/beta_masks/" + str(N_SNPS) + "_beta_mask.pkl","rb"))
    train_files, val_files = train_val_split(data_directory+'train/',n_train=48)
    # train
    for epoch in range(N_EPOCHS):
        train_iterator = get_dataloader(data_directory, ENCODING, 12, train_files, beta_mask)
        valid_iterator = get_dataloader(data_directory, ENCODING, 6, val_files, beta_mask)
        # TRAIN
        i = 0
        while i < len(train_files):
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
            while i < len(val_files):
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
                val_r2 += r2_score(y_pred.cpu().numpy().ravel(), Y.cpu().numpy().ravel())
                # suppress constant input warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    val_r += pearsonr(y_pred.ravel().cpu().numpy(), Y.ravel().cpu().numpy())[0]
                i += 1
        # Save a Ray Tune checkpoint & report score to Tune
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (model.state_dict(), optimiser.state_dict()), path)
        if not np.isnan(val_r):
            r = (val_r / i)
        else:
            r = -1000
        tune.report(r2=(val_r2 / i), loss=(val_loss / i), r=r)

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
    # free up any memory you can
    torch.cuda.empty_cache()

    # generate architectures
    layer_params = [
        [100,10,10],
        [100,2,2,2],
        [100,5,2,5,2],
        [100,2,2,5,5]
    ]
    architectures = []
    for r in layer_params:
        a = make_architecture(N_INPUTS, 1, r)
        architectures.append(a)

    bellot_architectures = [
        [N_INPUTS, 32, 1],
        [N_INPUTS, 64, 64, 1],
        [N_INPUTS, 32, 32, 32, 32, 32, 1],
        [N_INPUTS, 128, 128, 128, 128, 1]
    ]
    #architectures = bellot_architectures
    print("\nARCHITECTURE CHOICES")
    print(architectures)
    # define config
    config = {
        "arch": tune.grid_search(architectures),
        "activation": tune.grid_search(["ELU", "ReLU","LeakyReLU"]),#
        "dropout": tune.grid_search([0,0.1,0.2,0.3]),#
        "optim": tune.choice(["radam","adam","adamw","adamax",]), #"nadam","spadam","sgd","rmsprop",,
        "lr": tune.loguniform(1e-4, 1e-2),
        "loss": tune.grid_search(["MSE"])#,"huber"
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
        metric="r",
        mode="max",
        num_samples=1,
        scheduler=scheduler,
        max_concurrent_trials=3,

    )
    best_trial = result.get_best_trial("r", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation R: {}".format(
        best_trial.last_result["r"]))
    df = result.results_df
    sorted = df.sort_values('r', ascending=False)
    print("\n\n====================================================================\n")
    print(sorted)
    filename = "grid_search_mse/encoding" + str(ENCODING) + "_" + str(N_SNPS) + "_tuneresults.csv"
    sorted.to_csv(filename)

if __name__ == "__main__":
    main()






