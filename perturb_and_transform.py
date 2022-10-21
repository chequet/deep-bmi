import torch
import numpy as np
from numpy.random import default_rng
import math

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

def perturb_func(original_input: Tensor, n_interp_features: int, **_):
    perturbed_sample = torch.ones((n_interp_features,), dtype=torch.int8)
    # choose random number of elements to switch off, between 1 and all genes
    n = np.random.randint(1,n_interp_features)
    # choose n random indices to set to zero
    indices = default_rng().choice(np.arange(1,n_interp_features),size=n)
    perturbed_sample[[indices]] = 0
    return perturbed_sample.reshape(1,n_interp_features)

def from_interp_rep_transform(curr_sample: Tensor, original_input: Tensor, gene_index_array: np.array, **_):
    perturbed_sample = torch.clone(original_input)
    # get SNP indices for gene at each index
    for i in range(len(gene_index_array)):
        indices = gene_index_array[i]
        if curr_sample[0][i] == 0:
            # set corresponding SNPs to 0
            perturbed_sample[indices] = 0
        else:
            pass
    return perturbed_sample

def forward(inp, model):
    model.eval()
    X = inp.to(device)
    pred = model(X.float())
    return pred.detach().cpu()

def get_low_mse_samples(mse_cutoff, n_samples, bmi_lower, bmi_upper, groundtruths, errors):
    low_mse = np.array(errors < mse_cutoff)
    print(len(np.where(low_mse==True)[0]))
    bmi_low = np.array(groundtruths >= bmi_lower)
    print(len(np.where(bmi_low==True)[0]))
    bmi_high = np.array(groundtruths < bmi_upper)
    print(len(np.where(bmi_high==True)[0]))
    bmi_strat = bmi_low*bmi_high
    print(len(np.where(bmi_strat==True)[0]))
    low_mse_strat = bmi_strat*low_mse
    locs = np.where(low_mse_strat == True)[0]
    print(len(locs))
    np.random.shuffle(locs)
    return locs[:n_samples]

def get_sample(index, files, batch_size, fpath):
    # get batch number
    batch_i = math.floor(index/batch_size)
    print(batch_i)
    batch = files[batch_i]
    print(batch)
    filepath = fpath + batch
    b = np.load(filepath)
    # get array to extract
    array_i = index%batch_size
    print(array_i)
    array = b['x'][array_i]
    return array