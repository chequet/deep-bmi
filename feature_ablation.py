import sys
from BasicEmbeddedDataset import *
import pickle
import torch
import os
import numpy as np
from lime import get_test_set, get_masks

use_cuda = torch.cuda.is_available()

def single_gene_ablation(data, model, gene_keys, ordered_feature_masks, dict_file_name, lin_mod=False):
    device = torch.device("cuda:0" if (use_cuda and lin_mod==False) else "cpu")
    # data should be pre-filtered for BMI category and mse
    if not lin_mod:
        model.to(device)
    diffs_dict = {}
    for k in gene_keys:
        print(k)
        diffs = []
        mask = torch.tensor(ordered_feature_masks[k]).to(device)
        for i in range(len(data)):
            if lin_mod:
                og_inp = data[i].reshape(1, -1)
                og_pheno = model.predict(og_inp)
                new_inp = (og_inp * mask).reshape(1, -1)
                new_pheno = model.predict(new_inp)
                diff = (og_pheno - new_pheno)
            else:
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

def check_if_done(dict_directory, gene, linmod=False):
    if not linmod:
        dict_path = dict_directory + gene + "_pairs_dict.pkl"
    else:
        dict_path = dict_directory + "LIN_" + gene + "_pairs_dict.pkl"
    if os.path.exists(dict_path):
        return True, gene
    else:
        return False, None

def add_other_dict_keys(search_gene, dict_directory, linmod=False):
    files = os.listdir(dict_directory)
    if not linmod:
        files = [item for item in files if item[:3] != "LIN" and item[0].isupper()]
        og_path = dict_directory + search_gene + "_pairs_dict.pkl"
    else:
        og_path = dict_directory + "LIN_" + search_gene + "_pairs_dict.pkl"
        files = [item for item in files if item[:3] == "LIN"]
    og_dict = pickle.load(open(og_path, "rb"))
    for f in files:
        if not linmod:
            key = f.split("_")[0] + "_" + search_gene
        else:
            key = f.split("_")[1] + "_" + search_gene
        dict = pickle.load(open(dict_directory + f, "rb"))
        if key in dict.keys():
            new_key = search_gene + "_" + key.split("_")[0]
            og_dict[new_key] = dict[key]

def one_gene_pairwise(data, ordered_feature_masks, start_gene, gene_set, diffs_dict, model, dict_directory, lin_mod=False):
    # perturb given gene with comparison set and store perturbation results
    # comparison_set should be list of strings
    # data should be pre-filtered for BMI category and mse
    pairs_dict = {}
    if not lin_mod:
        dict_path = dict_directory + start_gene + "_pairs_dict.pkl"
    else:
        dict_path = dict_directory + "LIN_" + start_gene + "_pairs_dict.pkl"
    device = torch.device("cuda:0" if (use_cuda and lin_mod==False) else "cpu")
    gene_mask = ordered_feature_masks[start_gene]
    g = 1
    for gene in gene_set:
        print("gene %i of %i" % (g, len(gene_set)), end='\r')
        key = start_gene + "_" + gene
        mask = ordered_feature_masks[gene]
        joint_mask = torch.tensor(gene_mask * mask).to(device)
        diff_diffs = []
        c = 0
        for i in range(len(data)):
            linear_diff = diffs_dict[start_gene][c] + diffs_dict[gene][c]
            if lin_mod:
                og_inp = data[i].reshape(1, -1)
                og_pheno = model.predict(og_inp)
                new_inp = (og_inp * joint_mask).reshape(1, -1)
                new_pheno = model.predict(new_inp)
                pair_diff = (og_pheno - new_pheno)
            else:
                og_inp = data[i].to(device)
                og_pheno = model(og_inp.float())
                new_inp = og_inp * joint_mask
                new_pheno = model(new_inp.float())
                pair_diff = (og_pheno - new_pheno).detach().cpu().numpy()
            diff_diffs.append(pair_diff - linear_diff)
            c += 1
        pairs_dict[key] = np.mean(np.absolute(diff_diffs))
        g += 1
    print("writing pairs dict for %s to %s..." % (start_gene, dict_path))
    pickle.dump(pairs_dict, open(dict_path, "wb"))

def pairwise_ablation(data, ordered_feature_masks, gene_set,
                      diffs_dict, model, dict_directory, lin_mod=False):
    # exhaustive search of comparison set
    searched_genes = set()
    for start_gene in gene_set:
        print(start_gene)
        comparison_set = [g for g in gene_set if g not in searched_genes]
        one_gene_pairwise(data, ordered_feature_masks, comparison_set, diffs_dict,
                          model, dict_directory, lin_mod)
        searched_genes.add(start_gene)
    return True

def main(start_index):
    print("starting at index %i"%start_index)
    # initialise
    ordered_feature_masks = pickle.load(open("../gene_masks/10k_full_genes_ordered_feature_masks.pkl", "rb"))
    model = torch.load("10000radam_elu_0.2_huber4.pt")
    lin_model = pickle.load(open("10000_enc_4_best_SGDRegressor.pkl","rb"))
    print(model)
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
                                                                               False, 2, "py2"), **params))
    X_data = get_test_set(test_sample_loader, testfiles)
    # filter for BMI category, MSE
    mse_mask = np.array([1 if i < 0.1 else 0 for i in mses])
    joint_sample_mask = mse_mask * (np.array(obese_1_mask) + np.array(obese_2_mask))
    X_data_filtered = X_data[joint_sample_mask.astype(bool)]
    # print("beginning single gene ablation...")
    # lin_diffs = single_gene_ablation(X_data_filtered, model, gene_keys,
    #                                    ordered_feature_masks, "../diffs_dicts/linmod_diffs_dict.pkl", lin_mod=True)
    lin_diffs = pickle.load(open("../diffs_dicts/linmod_diffs_dict.pkl","rb"))
    lin_means = get_unsigned_means(lin_diffs, "../diffs_dicts/linmod_means_dict.pkl")
    diffs_dict = pickle.load(open("../diffs_dicts/obese12diffs.pkl","rb"))
    unsigned_means_dict = get_unsigned_means(diffs_dict, "../diffs_dicts/obese12means.pkl")
    sorted_unsigned_lin = sorted(lin_means.items(), key=lambda x: x[1], reverse=True)
    sorted_unsigned = sorted(unsigned_means_dict.items(), key=lambda x:x[1], reverse=True)
    # exhaustive search!
    print("beginning pairwise ablation...")
    genes = [tup[0] for tup in sorted_unsigned[start_index:]]
    pairwise_ablation(X_data_filtered, ordered_feature_masks, genes, diffs_dict, model, "../diffs_dicts/", lin_mod=False)

if __name__ == "__main__":
    main(start_index = sys.argv[1])
