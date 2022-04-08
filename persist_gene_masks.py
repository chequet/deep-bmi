import numpy as np
import pickle

N_SNPs = 10000
# make dictionary of gene names to SNP masks
gene_mask_dict = {}
rsid_10k = np.load('../10k_rsid.npy')
gene_dict = pickle.load(open('../gene_rsid_dict.pkl', 'rb'))
for key in gene_dict.keys():
    if key == key: # avoid nan key
        print(key)
        gene = gene_dict[key]
        snp_set = set(gene)
        print("getting indices...")
        gene_inds = []
        for rsid in snp_set:
            ind = np.where(rsid_10k == rsid)
            gene_inds.append(ind[0])
        gene_inds = np.concatenate(gene_inds)
        # convert into projection encoding format
        mask = np.ones((N_SNPs,2))
        print("populating mask...")
        for i in gene_inds:
            mask[i] = np.array([0,0])
        mask = mask.ravel()
        gene_mask_dict[key] = mask
pickle.dump(gene_mask_dict,open("gene_mask_dict.pkl","wb"))
