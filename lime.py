from perturb_and_transform import *
from BasicEmbeddedDataset import *
import pickle
import torch
import os
import numpy as np
from captum.attr import LimeBase
from captum._utils.models.linear_model import SkLearnLinearModel
from collections import OrderedDict

def get_masks(test_phenos):
    underweight_mask = [1 if p < -1.82 else 0 for p in test_phenos]
    healthy_mask = [1 if (p >= -1.82 and p < -0.44) else 0 for p in test_phenos]
    overweight_mask = [1 if (p >= -0.44 and p < 0.63) else 0 for p in test_phenos]
    obese_1_mask = [1 if (p >= 0.63 and p < 1.69) else 0 for p in test_phenos]
    obese_2_mask = [1 if (p >= 1.69 and p < 2.76) else 0 for p in test_phenos]
    obese_3_mask = [1 if (p >= -1.82 and p < -0.44) else 0 for p in test_phenos]
    return underweight_mask, healthy_mask, overweight_mask, obese_1_mask, obese_2_mask, obese_3_mask

def get_test_set(test_sample_loader, testfiles):
    X = []
    for i in range(len(testfiles)):
        print(i)
        batch = next(test_sample_loader)
        X.append(batch[0])
    X_data = torch.cat(X)
    return X_data

def get_attr_coefs(data, model, mask1, mask2, gene_keys, gene_mask_values, mses, lime_attr):
    attr_coef_matrix = []
    for i in range(len(data)):
        if (mask1[i] == 1 or mask2[i] == 1) and (mses[i] < 0.1):
            print("%i/%i"%(i,len(data)))
            inp = data[i]
            # do LIME
            attr_coefs = lime_attr.attribute(inp, additional_forward_args=[model], kernel_width=1.1,
                                     n_interp_features=len(gene_keys), gene_index_array=gene_mask_values,
                                     show_progress=False)
            # store results
            attr_coef_matrix.append(np.array(attr_coefs)[0])
    attr = np.array(attr_coef_matrix)
    return attr


def main():
    # get ordered keys and values for gene features with >1 SNP
    ordered_feature_masks = pickle.load(open("../gene_masks/10k_ordered_feature_masks.pkl","rb"))
    gene_keys = list(ordered_feature_masks.keys())
    gene_mask_values = np.array([ordered_feature_masks[key] for key in gene_keys])
    # get best 10k nn
    model = torch.load("10000radam_elu_0.2_huber4.pt")
    # first make a bmi mask for test set samples
    test_samples = pickle.load(open("../sample_sets/testset.pkl","rb"))
    pheno_dict = pickle.load(open("../phenotypes/scaled_phenotype_dict.pkl","rb"))
    test_phenos = [pheno_dict[s] for s in test_samples]
    underweight_mask, healthy_mask, overweight_mask, obese_1_mask, obese_2_mask, obese_3_mask = get_masks(test_phenos)
    mses = pickle.load(open("10000_test_mses.pkl", "rb"))
    # get entire X test dataset
    params = {'batch_size': None,
              'num_workers': 4}
    # no shuffle
    testfiles = os.listdir("../1000_data_relabelled/test/")
    test_sample_loader = iter(torch.utils.data.DataLoader(BasicEmbeddedDataset("../10000_data_relabelled/test/",
                                                                              testfiles,
                                                                               False, 2), **params))
    X_data = get_test_set(test_sample_loader, testfiles)
    split = int(np.ceil(len(X_data) / 2))
    print(split)
    X_data_1 = X_data[:split]
    X_data_2 = X_data[split:]
    # set up lime
    lime_attr = LimeBase(forward,
                         SkLearnLinearModel("linear_model.Ridge"),
                         similarity_func=similarity_kernel,
                         perturb_func=perturb_func,
                         perturb_interpretable_space=True,
                         from_interp_rep_transform=from_interp_rep_transform,
                         to_interp_rep_transform=None)
    # do lime for each subgroup
    print("subgroup 1")
    attr1 = get_attr_coefs(X_data_1, model, obese_1_mask, obese_2_mask, gene_keys, gene_mask_values, mses, lime_attr)
    np.save("obese12_bmi_lime_results1", attr1)
    print("subgroup 2")
    attr2 = get_attr_coefs(X_data_2, model, obese_1_mask, obese_2_mask, gene_keys, gene_mask_values, mses, lime_attr)
    np.save("obese12_bmi_lime_results2", attr2)

if __name__ == "__main__":
    main()
