# pipeline to generate new batches based on locke meta-analysis SNPs
# takes number of SNPs as input

import os
import sys
import numpy as np
from hail.linalg import BlockMatrix
from hail.utils import new_local_temp_file, local_path_uri
from functools import partial
import hail as hl
import hailtop
import pandas as pd
from MyIterableDataset import *


hl.init(default_reference='GRCh37', spark_conf={'spark.driver.memory': '400G'})


def to_numpy_patch(bm, _force_blocking=False):
    if bm.n_rows * bm.n_cols > 1 << 31 or _force_blocking:
        path = new_temp_file()
        bm.export_blocks(path, binary=True)
        return BlockMatrix.rectangles_to_numpy(path, binary=True)

    path = new_local_temp_file()
    try:
        uri = local_path_uri(path)
        bm.tofile(uri)
        return np.fromfile(path).reshape((bm.n_rows, bm.n_cols))
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def import_and_index(numbers):
    """ imports bgen files into hail
	numbers [] is the chromosome numbers to be imported and indexed """
    path2 = "../bgen/ukb_imp_chr"
    path3 = "_v3.bgen"
    path4 = "_v3.sample"
    ef = ['GT', 'dosage']
    # create a matrix to store the matrixtables
    chrs = []
    for i in numbers:
        # using the same sample file for all bgens to avoid confusion
        bgen_path = path2 + str(i) + path3
        print(bgen_path)
        sample_path = path2 + "1" + path4
        print(sample_path)
        print("\n\nimporting matrixtable for chromosome " + str(i) + "...")
        chr_i = hl.import_bgen(bgen_path, entry_fields=ef, sample_file=sample_path)
        chrs.append(chr_i)

    return chrs


def combine(chrs):
    """ join all matrixtables into single matrixtable using union_rows """
    print("\n\ncombining all chromosomes into single matrixtable...")
    # start with union of chrs1 and 2, then iterate to 22
    fullchr = chrs[0].union_rows(chrs[1])
    for i in range(2, len(chrs)):
        print("adding chromosome " + str(i + 1))
        fullchr = fullchr.union_rows(chrs[i])

    return fullchr

def make_block_matrix(mt, n_snps, write=True):
    AS_BLOCK_MATRIX_FILE = "blockmatrix_"+ str(n_snps) + ".mt"
    if write:
        hl.linalg.BlockMatrix.write_from_entry_expr(mt.GT.n_alt_alleles(),
                                                AS_BLOCK_MATRIX_FILE, overwrite=True, mean_impute=True)
    bm = hl.linalg.BlockMatrix.read(AS_BLOCK_MATRIX_FILE)
    GROUP_SIZE = 2048
    column_groups = hailtop.utils.grouped(GROUP_SIZE, list(range(mt.count_cols())))
    return bm, column_groups

def generate_data(iterable_dataset, n_snps):
    i = 1
    for X,Y in iterable_dataset:
        name = "../new_data_" + n_snps + "/batch" + str(i)
        print("saving %s..." % name)
        np.savez_compressed(name, x=X, y=Y)
        i += 1

def main(n_snps):
    print("updated paths!")
    print("generating data with %s SNPs"%n_snps)
    # read in bgens
    numbers = range(1, 23)
    chrs = import_and_index(numbers)
    full_mt = combine(chrs)
    # read in samples
    # try OLD SAMPLES to see what happens
    samples = np.load('../qc_samples.npy').tolist()
    # filter mt for samples
    sample_set = hl.literal(samples)
    full_mt = full_mt.filter_cols(sample_set.contains(full_mt.s))
    # make phenotype blocks with phenotype dictionary
    phenos_dict = pd.read_pickle('../pickles/new_pheno_dict.pkl')
    phenos = [[phenos_dict[s]] for s in samples]
    # read in SNPs, truncate for N
    top_snps = np.load(open('../new50k.npy','rb'))
    top_snps = top_snps[:int(n_snps)]
    print("top SNPs: %i"%len(top_snps))
    # filter mt for SNPs
    snp_set = hl.literal([hl.parse_locus(item) for item in top_snps])
    full_mt = full_mt.filter_rows(snp_set.contains(full_mt.locus))
    bm, column_groups = make_block_matrix(full_mt,n_snps)
    batch_list = list(column_groups)
    data = MyIterableDataset(bm, phenos, batch_list, n_snps, False)
    generate_data(data,n_snps)

if __name__ == "__main__":
    main(n_snps=sys.argv[1])
