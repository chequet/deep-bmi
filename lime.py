from perturb_and_transform import *
from BasicEmbeddedDataset import *
import pickle
import torch
import os
import numpy as np
from captum.attr import LimeBase
from captum._utils.models.linear_model import SkLearnLinearModel
from collections import OrderedDict


# get best 10k nn
model = torch.load("10000radam_elu_0.2_huber4.pt")

# get ordered keys and values for gene features with >1 SNP
ordered_feature_masks = pickle.load(open("../gene_masks/10k_ordered_feature_masks.pkl","rb"))
gene_keys = list(ordered_feature_masks.keys())
gene_mask_values = np.array([ordered_feature_masks[key] for key in gene_keys])

# aggregate gene stats for each bmi category
# first make a bmi mask for test set samples
test_samples = pickle.load(open("../sample_sets/testset.pkl","rb"))
pheno_dict = pickle.load(open("../phenotypes/scaled_phenotype_dict.pkl","rb"))
test_phenos = [pheno_dict[s] for s in test_samples]
#pickle.dump(test_phenos, open("../phenotypes/test_phenos.pkl","wb"))
underweight_mask = [1 if p < -1.82 else 0 for p in test_phenos]
healthy_mask = [1 if (p >= -1.82 and p < -0.44) else 0 for p in test_phenos]
overweight_mask = [1 if (p >= -0.44 and p < 0.63) else 0 for p in test_phenos]
obese_1_mask = [1 if (p >= 0.63 and p < 1.69) else 0 for p in test_phenos]
obese_2_mask = [1 if (p >= 1.69 and p < 2.76) else 0 for p in test_phenos]
obese_3_mask = [1 if (p >= -1.82 and p < -0.44) else 0 for p in test_phenos]

mses = pickle.load(open("10000_test_mses.pkl","rb"))

# get entire X test dataset
params = {'batch_size': None,
          'num_workers': 4}
# no shuffle
test_sample_loader = iter(torch.utils.data.DataLoader(BasicEmbeddedDataset("../10000_data_relabelled/test/",
                                                                           os.listdir("../10000_data_relabelled/test/"),
                                                                           False, 2), **params))
X = []
for i in range(len(os.listdir("../1000_data_relabelled/test/"))):
    print(i)
    batch = next(test_sample_loader)
    X.append(batch[0])
X_data = torch.cat(X)

split = int(np.ceil(len(X_data)/2))
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

# do lime for each group
print("subgroup 1")
attr_coef_matrix_1 = []
for i in range(len(test_samples)):
    if (obese_1_mask[i] == 1 or obese_2_mask[i] == 1) \
            and (mses[i] < 0.1):
        print("%i/%i"%(i,len(test_samples)))
        inp = X_data[i]
        # do LIME
        attr_coefs = lime_attr.attribute(inp, additional_forward_args=[model], kernel_width=1.1,
                                 n_interp_features=len(gene_keys), gene_index_array=gene_mask_values,
                                 show_progress=False)
        # store results
        attr_coef_matrix_1.append(np.array(attr_coefs)[0])
attr = np.array(attr_coef_matrix_1)
np.save("obese12_bmi_lime_results1", attr)

print("subgroup 2")
attr_coef_matrix_2 = []
for i in range(len(test_samples)):
    if (obese_1_mask[i] == 1 or obese_2_mask[i] == 1) \
            and (mses[i] < 0.1):
        print("%i/%i"%(i,len(test_samples)))
        inp = X_data[i]
        # do LIME
        attr_coefs = lime_attr.attribute(inp, additional_forward_args=[model], kernel_width=1.1,
                                 n_interp_features=len(gene_keys), gene_index_array=gene_mask_values,
                                 show_progress=False)
        # store results
        attr_coef_matrix_2.append(np.array(attr_coefs)[0])
attr = np.array(attr_coef_matrix_2)
np.save("obese12_bmi_lime_results2", attr)