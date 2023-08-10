from BasicEmbeddedDataset import *
import pickle
import torch
import os
import numpy as np
from lime import get_test_set, get_masks

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

def pairwise_ablation(data, ordered_feature_masks, comparison_set, diffs_dict, model):
    # perturb given gene with comparison set and store perturbation results
    # comparison_set should be list of strings
    # data should be pre-filtered for BMI category and mse
    searched_genes = set()
    pairs_dict = {}
    for start_gene in comparison_set:
        print(start_gene)
        comparison_subset = [g for g in comparison_set if (g!=start_gene and g not in searched_genes)]
        gene_mask = ordered_feature_masks[start_gene]
        print(len(comparison_subset))
        for gene in comparison_subset:
            key = start_gene + "_" + gene
            mask = ordered_feature_masks[gene]
            joint_mask = torch.tensor(gene_mask * mask).to(device)
            diff_diffs = []
            c = 0
            for i in range(len(data)):
                linear_diff = diffs_dict[start_gene][c] + diffs_dict[gene][c]
                og_inp = data[i].to(device)
                og_pheno = model(og_inp.float())
                new_inp = og_inp * joint_mask
                new_pheno = model(new_inp.float())
                pair_diff = (og_pheno - new_pheno).detach().cpu().numpy()
                diff_diffs.append(pair_diff - linear_diff)
                c += 1
            pairs_dict[key] = np.mean(np.absolute(diff_diffs))
        searched_genes.add(start_gene)
    return pairs_dict

def main():
    # initialise
    ordered_feature_masks = pickle.load(open("../gene_masks/10k_full_genes_ordered_feature_masks.pkl", "rb"))
    model = torch.load("10000radam_elu_0.2_huber4.pt")
    test_samples = pickle.load(open("../sample_sets/testset.pkl", "rb"))
    pheno_dict = pickle.load(open("../phenotypes/scaled_phenotype_dict.pkl", "rb"))
    test_phenos = [pheno_dict[s] for s in test_samples]
    underweight_mask, healthy_mask, overweight_mask, obese_1_mask, obese_2_mask, obese_3_mask = get_masks(test_phenos)
    gene_keys = list(ordered_feature_masks.keys())
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
    # filter for BMI category, MSE
    mse_mask = np.array([1 if i < 0.1 else 0 for i in mses])
    joint_sample_mask = mse_mask * (np.array(obese_1_mask) + np.array(obese_2_mask))
    X_data_filtered = X_data[joint_sample_mask.astype(bool)]
    diffs_dict = pickle.load(open("../diffs_dicts/obese12diffs.pkl","rb"))
    unsigned_means_dict = get_unsigned_means(diffs_dict, "../diffs_dicts/obese12means.pkl")
    sorted_unsigned = sorted(unsigned_means_dict.items(), key=lambda x: x[1], reverse=True)
    # exhaustive search!
    genes = [tup[0] for tup in sorted_unsigned[:20]]
    pairs_dict = pairwise_ablation(X_data_filtered, ordered_feature_masks, genes, diffs_dict, model)
    pickle.dump(pairs_dict, open("../diffs_dicts/top20_mean_pairs_dict.pkl","wb"))

if __name__ == "__main__":
    main()