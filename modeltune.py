### architecture gridsearch with ray tune
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from MyIterableDataset3 import *
from OneHotIterableDataset import *
from BasicEmbeddedDataset import *
import os
from modeltrain import train_val_split
from FlexibleNet import *

N_INPUTS = 999
N_EPOCHS = 10

# def cross_validation(data, k=5):
    # leave for now

def get_dataloaders(data_directory, type=3):
    train_files, val_files = train_val_split(data_directory + 'train/',n_train=100)
    n_train = len(train_files)
    n_val = len(val_files)
    trainparams = {'batch_size': None,
                   'num_workers': 4}
    valparams = {'batch_size': None,
                 'num_workers': 2}
    #TODO add code for other encodings
    if type == 1:
        pass
        # code for SVE
    elif type ==2:
        pass
        # code for one hot
    elif type == 3:
        train_iterator = iter(
            torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory + 'train/', train_files, True, 1),
                                        **trainparams))
        valid_iterator = iter(
            torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory + 'train/', val_files, True, 1),
                                        **valparams))
    return train_iterator, valid_iterator, n_train, n_val

def train(config, checkpoint_dir=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    # set up model according to config
    model = FlexibleNet(config["arch"],config["dropout"],config["activation"])
    model.to(device)
    loss_fn = torch.nn.MSELoss(reduction='mean')
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
    data_directory = "/data/old_data/" + str(N_INPUTS) + "_data/"
    # train
    for epoch in range(N_EPOCHS):
        train_iterator, valid_iterator, n_train, n_val = get_dataloaders(data_directory, type=3)
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
                i += 1
        # Save a Ray Tune checkpoint & report score to Tune
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (model.state_dict(), optimiser.state_dict()), path)

        tune.report(loss=(val_loss / i))
    print("done.")

def main():
    # define config
    config = {
        "arch": tune.grid_search([[1998,1000,100,10,1],
                                  [1998,200,20,2,1],
                                  [1998,100,100,100,10,1],
                                  [1998,1998,1998,100,10,1],
                                  [1998,1000,500,250,125,60,30,1],
                                  [1998,500,125,25,5,1]]),
        "activation": tune.grid_search(["ELU","ReLU","LeakyReLU"]),
        "dropout": tune.grid_search([0,0.1,0.2,0.3,0.4]),
        "optim": tune.grid_search(["adam","sgd","rmsprop","adamw","spadam","nadam","radam","adamax"]),
        "lr": tune.grid_search([1e-4,1e-3,1e-2,1e-1]),
    }
    scheduler = ASHAScheduler(
        max_t=N_EPOCHS,
        grace_period=1,
        reduction_factor=2)
    # run
    print("running...")
    result = tune.run(
        tune.with_parameters(train),
        resources_per_trial={"cpu": 3, "gpu": 0.25},
        config=config,
        metric="loss",
        mode="min",
        #num_samples=num_samples,
        scheduler=scheduler
    )
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

if __name__ == "__main__":
    main()






