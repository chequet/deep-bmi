# LIME interpretation of all samples

import numpy as np
import torch
import pickle
from captum.attr import Lime
import sys
from generators.BasicEmbeddedDataset import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)
og_model = torch.load('embed1_92.pt')


def forward(inp):
    og_model.eval()
    X = inp.to(device)
    pred = og_model(X.float())
    return pred


# stratify by BMI category
# returns the loci of samples of the desired category given the batch number
def strat_batch_by_bmi(sample_mask, cat, batchno):
    cat_samples = np.where(sample_mask==cat)[0].tolist()
    start = 0 + batchno*4096
    end = start + 4096
    batch_loci = np.array([i for i in cat_samples if i >= start and i < end])
    adjusted_batch_loci = batch_loci - start
    return adjusted_batch_loci


# stratify by BMI category, but only keep samples with RMSE below 0.4
# returns the loci of samples of the desired category given the batch number
def strat_batch_by_bmi_and_rmse(sample_mask, cat, rmse_mask, batchno):
    rmse_mask = rmse_mask[0].tolist()
    cat_samples = np.where(sample_mask==cat)[0].tolist()
    start = 0 + batchno*4096
    end = start + 4096
    bmi_loci = np.array([i for i in cat_samples if i >= start and i < end])
    adjusted_bmi_loci = bmi_loci - start
    rmse_loci = np.array([i for i in rmse_mask if i >= start and i < end])
    adjusted_rmse_loci = rmse_loci - start
    # keep only loci that are the right BMI category AND pass the rmse filter
    loci_intersect = np.intersect1d(adjusted_bmi_loci, adjusted_rmse_loci)
    return loci_intersect


def main(bmi_cat):
    print("interpreting samples for BMI category %i"%bmi_cat)
    params = {'batch_size': None,
              'num_workers': 6}
    data_iterator = iter(torch.utils.data.DataLoader(BasicEmbeddedDataset('./tst/', True, 1), **params))
    n_testbatch = 18
    lime = Lime(forward)
    feature_mask = pickle.load(open('pickles/mask100k.pkl', 'rb'))
    feature_mask = torch.tensor(feature_mask).long()
    sample_mask = pickle.load(open('pickles/bmi_mask.pkl', 'rb'))
    rmse_mask = pickle.load(open('pickles/rmse_mask.pkl', 'rb'))
    # generate array to store results
    n_cat = len(np.where(sample_mask == bmi_cat)[0])
    results_arrays = []
    batch_count = 0
    sample_count = 0
    while batch_count < n_testbatch:
        print("\n\nbatch count: %i"%batch_count)
        # get batch
        in_batch_count = 0
        batch = next(data_iterator)
        # get samples associated with this category
        loci = strat_batch_by_bmi_and_rmse(sample_mask, bmi_cat, rmse_mask, batch_count)
        results_array = np.zeros([len(loci), 22])
        print("SAMPLES IN BATCH %i"%len(loci))
        for locus in loci:
            # do lime, add results to category array
            print("locus: %i"%locus)
            sample = batch[0][locus]
            attr = lime.attribute(sample, feature_mask=feature_mask, show_progress=False, return_input_shape=False)
            row = attr[0].numpy()
            results_array[in_batch_count] = row
            in_batch_count += 1
        results_arrays.append(results_array)
        batch_count += 1
    results_filename = "interp_results/rmse_filter_bmi_cat"+str(bmi_cat)+'.pkl'
    pickle.dump(results_arrays,open(results_filename,'wb'))


if __name__ == "__main__":
    main(bmi_cat=int(sys.argv[1]))