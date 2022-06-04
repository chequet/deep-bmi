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
import os, re, sys
from modeltrain import train_val_split
from FlexibleNet import *

N_INPUTS = 1000
N_EPOCHS = 10

# def cross_validation(data, k=5):
    # leave for now

def get_dataloaders(data_directory, type=3):
    train_files, val_files = train_val_split(data_directory + '/train/',n_train=100)
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
            torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory + '/train/', train_files, True, 1),
                                        **trainparams))
        valid_iterator = iter(
            torch.utils.data.DataLoader(BasicEmbeddedDataset(data_directory + '/train/', val_files, True, 1),
                                        **valparams))
    return train_iterator, valid_iterator

def train(config, checkpoint_dir=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    # set up model according to config
    model = FlexibleNet(config["arch"],config["dropout"],config["activation"])
    loss_fn = torch.nn.MSELoss(reduction='mean')
    learning_rate = config['lr']
    # choose optimiser based on config
    if config["optim"] == 'adam':
        optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    # TODO add other optimiser options!
    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimiser.load_state_dict(optimizer_state)
    data_directory = "../old_data/" + str(N_INPUTS) + "_data/"
    train_iterator, valid_iterator = get_dataloaders(data_directory, type=config['enc'])
    # train
    for epoch in range(N_EPOCHS):
        # TRAIN
        for i, batch in enumerate(train_iterator,0):
            print("batch index %i" % i, end='\r')
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
        # VALIDATE
        with torch.no_grad():
            val_loss = 0.0
            val_steps = 0
            for i, batch in enumerate(valid_iterator,0):
                print("validation batch index %i" % i, end='\r')
                X = batch[0].to(device)
                Y = batch[1].to(device)
                model.eval()
                # forward pass
                y_pred = model(X.float())
                # compute and print loss
                loss = loss_fn(y_pred, Y)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        # Save a Ray Tune checkpoint & report score to Tune
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (model.state_dict(), optimiser.state_dict()), path)

        tune.report(loss=(val_loss / val_steps))
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
        "activation": tune.choice(["ELU","ReLU","LeakyReLU"]),
        "dropout": tune.quniform(0, 0.4, 0.1),
        "optim": tune.choice(["adam","other"]),
        "lr": tune.loguniform(1e-4, 1e-1),
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






