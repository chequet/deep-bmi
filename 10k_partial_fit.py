import numpy as np
import sklearn
from sklearn.linear_model import SGDRegressor
from MyIterableDataset3 import *
from BasicEmbeddedDataset import *
from OneHotIterableDataset import *
from sklearn.model_selection import GridSearchCV
import torch
import math
from sklearn.metrics import mean_squared_error, r2_score

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

# SGDRegressor gridsearch
param_grid = {
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
}
sgd = SGDRegressor(early_stopping=True)
clf = GridSearchCV(sgd, param_grid, scoring=['neg_mean_squared_error', 'r2'], refit='neg_mean_squared_error')
DATA_DIR = '../10000_data/test/'
FILES = os.listdir(DATA_DIR)
params = {'batch_size': None,
          'num_workers': 1}  # using one worker as it doesn't take that long anyway

test_iterator = iter(torch.utils.data.DataLoader(BasicEmbeddedDataset(DATA_DIR, FILES, False, 2), **params))
print("generating data...")
# get test set
data = []
target = []
i = 0
while i < len(FILES):
    batch = next(test_iterator)
    X = batch[0].to(device)
    Y = batch[1].to(device)
    data.append(X.cpu().numpy())
    target.append(Y.cpu().numpy())
    i += 1
data = np.concatenate(data)
target = np.concatenate(target).ravel()

print("fitting classifier...")
clf.fit(data, target)

print("writing...")
with open('10k_p2_SGDRegressor.txt', 'w') as f:
    f.write("STANDARD VARIABLE ENCODING\n")
    f.write("best params: ")
    f.write(str(clf.cv_results_['params'][clf.best_index_]))
    f.write("\nbest neg mse: ")
    f.write(str(clf.cv_results_['mean_test_neg_mean_squared_error'][clf.best_index_]))
    f.write("\nbest R: ")
    f.write(str(math.sqrt((-1 * clf.cv_results_['mean_test_r2'][clf.best_index_]))))
f.close()
