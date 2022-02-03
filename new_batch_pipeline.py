# pipeline to generate new batches based on locke meta-analysis SNPs
# takes number of SNPs as input

import os
import numpy as np
from hail.linalg import BlockMatrix
from hail.utils import new_local_temp_file, local_path_uri
from functools import partial
import hail as hl
import hailtop
import pickle
import pandas as pd
import numpy as np
import pickle

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
    path2 = "bgen/ukb_imp_chr"
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


def main(n_snps):
    # read in bgens
    numbers = range(1, 23)
    chrs = import_and_index(numbers)
    full_mt = combine(chrs)
    # read in samples
    train = pickle.load(open('pickles/trainset.pkl', 'rb'))
    test = pickle.load(open('pickles/testset.pkl', 'rb'))
    samples = train + test
    # filter mt for samples
    sample_set = hl.literal(samples)
    full_mt = full_mt.filter_cols(sample_set.contains(full_mt))
    # read in SNPs, truncate for N
    top_snps = np.load(open('new50k.npy','rb'))
    top_snps = top_snps[:n_snps]


if __name__ == "__main__":
    main(n_snps=sys.argv[1])
