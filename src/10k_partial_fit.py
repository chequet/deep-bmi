from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import torch
from modeltune import *
import pickle as pkl

N_SNPS = sys.argv[1]
ENC = sys.argv[2]
print(N_SNPS, ENC)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

# SGDRegressor gridsearch
param_grid = {
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'epsilon' : [0.1, 0.5, 0.75, 1]
}
sgd = SGDRegressor(early_stopping=True)
clf = GridSearchCV(sgd, param_grid, scoring=['r2', 'neg_mean_squared_error'], refit='r2')
DATA_DIR = '../' + str(N_SNPS) + '_data/'
FILES = os.listdir(DATA_DIR)
params = {'batch_size': None,
          'num_workers': 1}  # using one worker as it doesn't take that long anyway

train_iterator, val_iterator, n_train, n_val = get_dataloaders(DATA_DIR, int(ENC), trainworkers=1, valworkers=1, n_train=len(FILES))
print("generating data...")
# get test set
data = []
target = []
i = 0
while i < len(FILES):
    batch = next(train_iterator)
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
fname = '../linear_gridsearch_results/' + str(N_SNPS) + "_encoding_" + str(ENC)
best_fname = fname + '_SGDRegressor.txt'
with open(best_fname, 'w') as f:
    f.write("STANDARD VARIABLE ENCODING\n")
    f.write("best params: ")
    f.write(str(clf.cv_results_['params'][clf.best_index_]))
    f.write("\nbest neg mse: ")
    f.write(str(clf.cv_results_['mean_test_neg_mean_squared_error'][clf.best_index_]))
    f.write("\nbest R: ")
    f.write(str(math.sqrt((-1 * clf.cv_results_['mean_test_r2'][clf.best_index_]))))
f.close()
results_fname = fname + "_resultsCV.pkl"
with open(results_fname, 'wb') as f:
    pkl.dump(clf.cv_results_, f)