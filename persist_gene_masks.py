import numpy as np
import pickle

N_SNPs = 10000
# make dictionary of gene names to SNP masks
gene_mask_dict = {}
rsid_10k = pickle.load(open('../new_rsid_top10k.pkl','rb'))
rsid_10k = np.array(rsid_10k)
gene_dict = pickle.load(open('../gene_rsid_dict2.pkl', 'rb'))
for key in gene_dict.keys():
    # avoid nan key
    if key == key:
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
pickle.dump(gene_mask_dict,open(str(N_SNPs) + "multi-snp_gene_mask_dict.pkl","wb"))
