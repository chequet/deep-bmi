import numpy as np
import torch
import pickle
from captum.attr import Lime
from BasicEmbeddedDataset import *
from sklearn.metrics import mean_squared_error
import math
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

def forward(inp, og_model):
    og_model.eval()
    X = inp.to(device)
    pred = og_model(X.float())
    return pred.cpu()

def main():
    # calculate RMSE for each sample
    og_model = torch.load('embed1_92.pt')
    params = {'batch_size': None,
              'num_workers': 6}
    data_iterator = iter(torch.utils.data.DataLoader(BasicEmbeddedDataset('./tst/', True, 1), **params))
    n_testbatch = 18
    batch_count = 0
    rmses = []
    while batch_count < n_testbatch:
        print("batch count: %i" % batch_count)
        # get batch
        batch = next(data_iterator)
        X = batch[0]
        Y = batch[1].detach().numpy()
        # forward pass
        y_pred = forward(X,og_model).detach().numpy()
        for i in range(len(Y)):
            mse = mean_squared_error(Y[i], y_pred[i])
            rmse = math.sqrt(mse)
            rmses.append(rmse)
        batch_count += 1
    pickle.dump(rmses,open('pickles/rmses.pkl','wb'))

if __name__ == "__main__":
    main()