import numpy as np
import sklearn
from sklearn.linear_model import SGDRegressor
from MyIterableDataset3 import *
from BasicEmbeddedDataset import *
from OneHotIterableDataset import *
from sklearn.model_selection import GridSearchCV
import torch
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import sys
import pickle

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

ENC = int(sys.argv[1])

# SGDRegressor gridsearch
param_grid = {
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
}

N_SNPS = [100, 500]
for N in N_SNPS:
    DATA_DIR = "../" + str(N) + "_data_relabelled/train/"
    print(DATA_DIR)
    FILES = os.listdir(DATA_DIR)
    trainfiles = FILES[:45]
    valfiles = FILES[45:]
    params = {'batch_size': None,
              'num_workers': 5}

    if ENC == 1:
        train_iterator = iter(torch.utils.data.DataLoader(MyIterableDataset(DATA_DIR, trainfiles, False), **params))
        valid_iterator = iter(torch.utils.data.DataLoader(MyIterableDataset(DATA_DIR, valfiles, False), **params))
    elif ENC == 2:
        train_iterator = iter(torch.utils.data.DataLoader(OneHotIterableDataset(DATA_DIR, trainfiles, False), **params))
        valid_iterator = iter(torch.utils.data.DataLoader(OneHotIterableDataset(DATA_DIR, valfiles, False), **params))
    elif ENC == 3:
        train_iterator = iter(
            torch.utils.data.DataLoader(BasicEmbeddedDataset(DATA_DIR, trainfiles, False, 1), **params))
        valid_iterator = iter(torch.utils.data.DataLoader(BasicEmbeddedDataset(DATA_DIR, valfiles, False, 1), **params))
    elif ENC == 4:
        train_iterator = iter(
            torch.utils.data.DataLoader(BasicEmbeddedDataset(DATA_DIR, trainfiles, False, 2), **params))
        valid_iterator = iter(torch.utils.data.DataLoader(BasicEmbeddedDataset(DATA_DIR, valfiles, False, 2), **params))

    # get data set
    data = []
    target = []
    i = 0
    while i < len(trainfiles):
        batch = next(train_iterator)
        X = batch[0].to(device)
        Y = batch[1].to(device)
        data.append(X.cpu().numpy())
        target.append(Y.cpu().numpy())
        i += 1
    data = np.concatenate(data)
    target = np.concatenate(target).ravel()

    # fit regressor
    sgd = SGDRegressor(early_stopping=True, verbose=3)
    clf = GridSearchCV(sgd, param_grid, scoring='neg_mean_squared_error')
    clf.fit(data, target)
    print(clf.cv_results_['params'][clf.best_index_])
    # test best regressor
    best_regressor = clf.best_estimator_
    # get test set
    val_data = []
    val_target = []
    i = 0
    while i < len(valfiles):
        batch = next(valid_iterator)
        X = batch[0].to(device)
        Y = batch[1].to(device)
        val_data.append(X.cpu().numpy())
        val_target.append(Y.cpu().numpy())
        i += 1
    val_data = np.concatenate(val_data)
    val_target = np.concatenate(val_target).ravel()
    val_pred = best_regressor.predict(val_data)
    r2 = r2_score(val_pred, val_target)
    pearsonr = stats.pearsonr(val_pred, val_target)
    mse = mean_squared_error(val_pred, val_target)
    print(r2)
    print(pearsonr)
    print(mse)
    print("writing results to txt file...")
    with open(str(N) + '_enc_' + str(ENC) + '_SGDRegressor.txt', 'w') as f:
        f.write("ENCODING" + str(ENC) + "\n")
        f.write("best params: ")
        f.write(str(clf.cv_results_['params'][clf.best_index_]))
        f.write("\nvalidation r2: ")
        f.write(str(r2))
        f.write("\nvalidation r: ")
        f.write(str(pearsonr))
        f.write("\nvalidation mse: ")
        f.write(str(mse))
    f.close()
    pickle.dump(best_regressor, open(str(N) + '_enc_' + str(ENC) + '_best_SGDRegressor.pkl', 'wb'))
