import pickle
import torch
import os
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def single_gene_ablation(data, model, gene_keys, ordered_feature_masks, dict_file_name):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # data should be pre-filtered for BMI category and mse
    model.to(device)
    diffs_dict = {}
    for k in gene_keys:
        print(k)
        diffs = []
        mask = torch.tensor(ordered_feature_masks[k]).to(device)
        for i in range(len(data)):
            og_inp = data[i].to(device)
            og_pheno = model(og_inp.float())
            new_inp = og_inp * mask
            new_pheno = model(new_inp.float())
            diff = (og_pheno - new_pheno).detach().cpu().numpy()
            diffs.append(diff)
        diffs_dict[k] = np.concatenate(diffs)
    # persist diffs dict
    pickle.dump(diffs_dict, open(dict_file_name, "wb"))
    return diffs_dict

def get_unsigned_means(diffs_dict, means_dict_path):
    unsigned_means_dict = {}
    for key in diffs_dict.keys():
        unsigned_means_dict[key] = np.mean(np.absolute(diffs_dict[key]))
    pickle.dump(unsigned_means_dict, open(means_dict_path, "wb"))
    return unsigned_means_dict

def pairwise_ablation(gene_name, data, ordered_feature_masks, comparison_set, diffs_dict, model):
    # perturb given gene with comparison set and store perturbation results
    # comparison_set should be ordered (gene name, score) tuples
    # data should be pre-filtered for BMI category and mse
    gene_mask = ordered_feature_masks[gene_name]
    pairs_dict = {}
    for gene in comparison_set:
        key = gene[0]
        mask = ordered_feature_masks[key]
        joint_mask = torch.tensor(gene_mask * mask).to(device)
        diff_diffs = []
        gene_diff = diffs_dict[gene_name][c]
        c = 0
        for i in range(len(data)):
            linear_diff = gene_diff + diffs_dict[key][c]
            og_inp = data[i].to(device)
            og_pheno = model(og_inp.float())
            new_inp = og_inp * joint_mask
            new_pheno = model(new_inp.float())
            pair_diff = (og_pheno - new_pheno).detach().cpu().numpy()
            diff_diffs.append(pair_diff - linear_diff)
            c += 1
        pairs_dict[key] = np.mean(np.absolute(diff_diffs))
    return pairs_dict
