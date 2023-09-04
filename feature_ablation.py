import sys
from BasicEmbeddedDataset import *
import pickle
import torch
import os
import numpy as np
from lime import get_test_set, get_masks
import math
from multiprocessing import Process

CUDA_VISIBLE_DEVICES=sys.argv[3]
N_CPUs = 8
use_cuda = torch.cuda.is_available()
if sys.argv[3] == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:"+sys.argv[3] if(use_cuda) else "cpu")

def single_gene_ablation(data, model, gene_keys, ordered_feature_masks, dict_file_name, lin_mod=False):
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

def one_gene_pairwise(data, ordered_feature_masks, start_gene, gene_set,
                      diffs_dict, model, dict_directory, lin_mod=False):
    # perturb given gene with comparison set and store perturbation results
    # comparison_set should be list of strings
    # data should be pre-filtered for BMI category and mse
    pairs_dict = {}
    if not lin_mod:
        dict_path = dict_directory + start_gene + "_pairs_dict.pkl"
    else:
        dict_path = dict_directory + "LIN_" + start_gene + "_pairs_dict.pkl"
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

def one_gene_parallel_pairwise(data, ordered_feature_masks, start_gene, gene_set,
                      diffs_dict, model, dict_directory, n_cpus, lin_mod=False):
    # divide gene set up into n_cpus batches
    miniset_size = int(math.ceil(len(gene_set)/n_cpus))
    print("miniset size: %i"%miniset_size)
    minisets = []
    for i in range(0,len(gene_set),miniset_size):
        minisets.append(gene_set[i:i+miniset_size])
    procs = []
    for miniset in minisets:
        proc=Process(target=one_gene_pairwise, args=(data, ordered_feature_masks, start_gene, miniset,
                      diffs_dict, model, dict_directory, lin_mod,))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()

def pairwise_ablation(data, ordered_feature_masks, gene_set,
                      diffs_dict, stop_gene, model, dict_directory, lin_mod=False, parallel=False):
    # exhaustive search of comparison set
    searched_genes = set()
    for start_gene in gene_set:
        print(start_gene)
        if start_gene == stop_gene:
            print("reached stop gene: %s"%stop_gene)
            return True
        searched_genes.add(start_gene)
        comparison_set = [g for g in gene_set if g not in searched_genes]
        if parallel:
            one_gene_parallel_pairwise(data, ordered_feature_masks, start_gene, comparison_set, diffs_dict,
                              model, dict_directory, N_CPUs, lin_mod)
        else:
            one_gene_pairwise(data, ordered_feature_masks, start_gene, comparison_set, diffs_dict,
                          model, dict_directory, lin_mod)
    return True

def check_overlap(gene1, gene2, gene_feature_mask):
    mask1 = gene_feature_mask[gene1]
    mask2 = gene_feature_mask[gene2]
    set1 = set(np.where(mask1==0)[0])
    set2 = set(np.where(mask2==0)[0])
    if len(set1.intersection(set2)) > 0:
        return True
    else:
        return False

def main(start_index, stop_index, lin):
    if lin==0:
        linmod = False
    elif lin==1:
        linmod = True
    print("starting at index %i and stopping at index %i"%(start_index, stop_index))
    # initialise
    print("DEVICE")
    print(device)
    ordered_feature_masks = pickle.load(open("../gene_masks/10k_full_genes_ordered_feature_masks.pkl", "rb"))
    if linmod:
        model = pickle.load(open("10000_enc_4_best_SGDRegressor.pkl","rb"))
    else:
        model = torch.load("10000radam_elu_0.2_huber4.pt")
        model.to(device)
    print(model)
    test_samples = pickle.load(open("../sample_sets/testset.pkl", "rb"))
    pheno_dict = pickle.load(open("../phenotypes/scaled_phenotype_dict.pkl", "rb"))
    test_phenos = [pheno_dict[s] for s in test_samples]
    underweight_mask, healthy_mask, overweight_mask, obese_1_mask, obese_2_mask, obese_3_mask = get_masks(test_phenos)
    gene_keys = list(ordered_feature_masks.keys())
    mses = pickle.load(open("10000_test_mses.pkl", "rb"))
    # get entire X test dataset
    params = {'batch_size': None,
              'num_workers': 1}
    # no shuffle
    testfiles = os.listdir("../10000_data_relabelled/test/")
    test_sample_loader = iter(torch.utils.data.DataLoader(BasicEmbeddedDataset("../10000_data_relabelled/test/",
                                                                               testfiles,
                                                                               False, 2, "y"), **params))
    X_data = get_test_set(test_sample_loader, testfiles)
    # filter for BMI category, MSE
    mse_mask = np.array([1 if i < 0.1 else 0 for i in mses])
    joint_sample_mask = mse_mask * (np.array(obese_1_mask) + np.array(obese_2_mask))
    X_data_filtered = X_data[joint_sample_mask.astype(bool)]
    # print("beginning single gene ablation...")
    # lin_diffs = single_gene_ablation(X_data_filtered, model, gene_keys,
    #                                    ordered_feature_masks, "../diffs_dicts/linmod_diffs_dict.pkl", lin_mod=True)
    # lin_diffs = pickle.load(open("../diffs_dicts/linmod_diffs_dict.pkl","rb"))
    # lin_means = get_unsigned_means(lin_diffs, "../diffs_dicts/linmod_means_dict.pkl")
    if linmod:
        diffs_dict = pickle.load(open("../diffs_dicts/linmod_diffs_dict.pkl","rb"))
        unsigned_means_dict = get_unsigned_means(diffs_dict, "../diffs_dicts/linmod_means_dict.pkl")
    else:
        diffs_dict = pickle.load(open("../diffs_dicts/obese12diffs.pkl","rb"))
        unsigned_means_dict = get_unsigned_means(diffs_dict, "../diffs_dicts/obese12means.pkl")
    # sorted_unsigned_lin = sorted(lin_means.items(), key=lambda x: x[1], reverse=True)
    sorted_unsigned = sorted(unsigned_means_dict.items(), key=lambda x:x[1], reverse=True)
    # exhaustive search!
    print("beginning pairwise ablation...")
    genes = [tup[0] for tup in sorted_unsigned[start_index:]]
    stop_gene = sorted_unsigned[stop_index][0]
    pairwise_ablation(X_data_filtered, ordered_feature_masks, genes, diffs_dict, stop_gene, model,
                      "../diffs_dicts/", lin_mod=linmod, parallel=False)

if __name__ == "__main__":
    main(start_index = int(sys.argv[1]), stop_index = int(sys.argv[2]), lin=int(sys.argv[4]))
