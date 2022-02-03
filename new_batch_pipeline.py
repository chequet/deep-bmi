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
hl.init(default_reference='GRCh37',spark_conf={'spark.driver.memory': '400G'})

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

def main(n_snps):


if __name__ == "__main__":
    main(n_snps=sys.argv[1])
