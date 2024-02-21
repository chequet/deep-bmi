# runs a gwas on hard call data using only data from training set
# takes chromosome range from the command line
# saves p values and stats in gwas_out

import hail as hl
import sys
import pickle
hl.init(default_reference='GRCh37',spark_conf={'spark.driver.memory': '250G'})  

from gtgwas import importchrs, annotate_gt, filter_variants_gt
from gwas.gwas_full2 import get_PCA, gwas

startchr = sys.argv[1]
endchr = sys.argv[2]
numbers = range(int(startchr),int(endchr))

# import chromosomes
chrs = importchrs(numbers)

for i in range(len(chrs)):
	# get current chromosome from chrs
	currentchr = chrs[i]
	currentnum = numbers[i]
	print("\n\n--CHROMOSOME %i--\n\n"%currentnum)
	# annotate
	currentchr = annotate_gt(currentchr, numbers)
	# select only training set (QC'd)
	print("getting training set for chromosome %i" %(currentnum))
	pickle_in = open("pickles/trainset.pkl","rb")
	trainset = pickle.load(pickle_in)
	trainset = hl.literal(trainset)
	currentchr = currentchr.filter_cols(trainset.contains(currentchr.s))
	# variant QC
	print("now performing variant QC for chromosome %i" %currentnum)
	currentchr = filter_variants_gt(currentchr)
	# gwas 
	currentchr = get_PCA(currentchr)
	print("now performing training set gwas for chromosome %i" %(currentnum))
	thisgwas = gwas(currentchr, True, False)
	stats = thisgwas.drop('n','y_transpose_x', 'sum_x')
	print("saving gwas stats...")
	filename = "gwas_out/chr" + str(currentnum) + "_stats.tsv.bgz"
	stats.export(filename)



