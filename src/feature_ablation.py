import sys
from generators.BasicEmbeddedDataset import *
import pickle
import torch
import os
import numpy as np
from lime import get_test_set, get_masks
import math

N_CPUs = 8

def single_gene_ablation(data, model, gene_keys, ordered_feature_masks, dict_file_name, device, lin_mod=False):
    # data should be pre-filtered for BMI category and mse
    if not lin_mod:
        model.to(device)
        model.eval()
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
    return og_dict

def one_gene_pairwise(data, ordered_feature_masks, start_gene, gene_set,
                      diffs_dict, model, dict_directory, device, lin_mod=False):
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
                model.eval()
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
                      diffs_dict, stop_gene, model, dict_directory, device, lin_mod=False):
    # exhaustive search of comparison set
    searched_genes = set()
    for start_gene in gene_set:
        print(start_gene)
        if start_gene == stop_gene:
            print("reached stop gene: %s"%stop_gene)
            return True
        searched_genes.add(start_gene)
        comparison_set = [g for g in gene_set if g not in searched_genes]
        one_gene_pairwise(data, ordered_feature_masks, start_gene, comparison_set, diffs_dict,
                          model, dict_directory, device, lin_mod)
    return True

def check_overlap(gene1, gene2, gene_feature_masks):
    # check if two genes have SNPs in common
    mask1 = gene_feature_masks[gene1]
    mask2 = gene_feature_masks[gene2]
    set1 = set(np.where(mask1==0)[0])
    set2 = set(np.where(mask2==0)[0])
    if len(set1.intersection(set2)) > 0:
        return True
    else:
        return False

def check_second_degree_overlap(gene1, gene2, gene_feature_masks, comparison_set):
    # check if the two input genes have overlapping genes in common
    for gene3 in comparison_set:
        if (check_overlap(gene1, gene3, gene_feature_masks) and
            check_overlap(gene2, gene3, gene_feature_masks)):
            return True, gene3
    return False, None

# more general methods for larger numbers of genes
def get_linear_diff(gene_set, index, diffs_dict):
    lin_diff = 0
    for gene in gene_set:
        lin_diff += diffs_dict[gene][index]
    return lin_diff

def get_joint_mask(gene_set, feature_masks):
    masks = []
    for gene in gene_set:
        m = feature_masks[gene]
        masks.append(m)
    joint_mask = masks[0]
    for i in range(1, len(masks)):
        joint_mask = joint_mask*masks[i]
    return joint_mask

# check top pairs for direction, plots
def single_pair_analysis(pair, model, data, joint_mask, device, diffs_dict):
    diff_diffs = []
    lin_diffs = []
    model_diffs = []
    c = 0
    model.eval()
    for i in range(len(data)):
        print("sample %i of %i" % (i, len(data)), end='\r')
        linear_diff = get_linear_diff(pair, c, diffs_dict)
        lin_diffs.append(linear_diff)
        og_inp = data[i].to(device)
        og_pheno = model(og_inp.float())
        new_inp = og_inp * joint_mask
        new_pheno = model(new_inp.float())
        model_diff = (og_pheno - new_pheno).detach().cpu().numpy()
        model_diffs.append(model_diff)
        diff_diffs.append(model_diff-linear_diff)
        c += 1
    return lin_diffs, model_diffs, diff_diffs

def main(start_index, stop_index, gpu, lin):
    if gpu == "-1":
        device = torch.device("cpu")
    else:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:" + gpu if (use_cuda) else "cpu")
    if lin==0:
        linmod = False
    elif lin==1:
        linmod = True
    print("starting at index %i and stopping at index %i"%(start_index, stop_index))
    # initialise
    print("DEVICE")
    print(device)
    ordered_feature_masks = pickle.load(open("gene_feature_masks_filtered.pkl","rb"))
    if linmod:
        model = pickle.load(open("10000_enc_4_best_SGDRegressor.pkl","rb"))
    else:
        model = torch.load("10000radam_elu_0.2_huber4.pt")
        model.to(device)
        model.eval()
    print(model)
    # load selected samples
    X_data_filtered = torch.tensor(np.load("mini_ablation_test_set.npz")['x'])
    if linmod:
        diffs_dict = pickle.load(open("../ablation_results/lin_mini_x_diffs.pkl","rb"))
        unsigned_means_dict = get_unsigned_means(diffs_dict, "../ablation_results/linear_miniset_means.pkl")
    else:
        diffs_dict = pickle.load(open("../ablation_results/mini_x_set_diffs.pkl","rb"))
        unsigned_means_dict = pickle.load(open("../ablation_results/miniset_means_dict.pkl", "rb"))
    # sorted_unsigned_lin = sorted(lin_means.items(), key=lambda x: x[1], reverse=True)
    sorted_unsigned = sorted(unsigned_means_dict.items(), key=lambda x:x[1], reverse=True)
    # exhaustive search!
    print("beginning pairwise ablation...")
    genes = [tup[0] for tup in sorted_unsigned if tup[0] in set(ordered_feature_masks.keys())]
    gene_set = genes[start_index:]
    stop_gene = genes[stop_index]
    print("stop gene: %s" %stop_gene)
    pairwise_ablation(X_data_filtered, ordered_feature_masks, gene_set, diffs_dict, stop_gene, model,
                      "../ablation_results/", device, lin_mod=linmod)
    # persist model to see if weights have changed
    # torch.save(model, "post_model.pt")

if __name__ == "__main__":
    main(start_index = int(sys.argv[1]), stop_index = int(sys.argv[2]), gpu = sys.argv[3], lin=int(sys.argv[4]))
