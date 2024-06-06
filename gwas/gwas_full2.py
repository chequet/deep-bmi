# a version of the full gwas designed to be called as individual functions within ipython 
import hail as hl
import pandas as pd
import datetime as d
# hl.init(default_reference='GRCh37',spark_conf={'spark.driver.memory': '250G'})  
from bokeh.plotting import output_file, save
from pprint import pprint
import subprocess
import os.path

def import_and_index(numbers):
	""" imports bgen files into hail 
	numbers [] is the chromosome numbers to be imported and indexed """
	path2 = "ukb_imp_chr"
	path3 = "_v3.bgen"
	path4 = "_v3.sample"
	ef = ['GT','dosage']
	# create a matrix to store the matrixtables
	chrs = []
	for i in numbers:
		# using the same sample file for all bgens to avoid confusion 
		bgen_path = path2 + str(i) + path3
		print(bgen_path)
		sample_path = path2 + "1" + path4
		print(sample_path)
		print("\n\nimporting matrixtable for chromosome " + str(i) + "...")
		# create index and contig recode if index file doesn't already exist
		if os.path.exists(bgen_path+".idx2"):
			print("index already exists for " + bgen_path)
		else:
			print("creating SNP index for " + bgen_path)
			new_contig = str(i)[0]
			current_contig = "0"+new_contig
			hl.index_bgen(bgen_path,contig_recoding={current_contig:new_contig})
		chr_i = hl.import_bgen(bgen_path, entry_fields=ef,sample_file=sample_path)
		chrs.append(chr_i)
		# subprocess.run(["rm",bgen_path])
		# subprocess.run(["rm",sample_path])

	return chrs

def combine(chrs, numbers):
	""" join all matrixtables into single matrixtable using union_rows """
	print("\n\ncombining all chromosomes into single matrixtable...")
	# start with union of chrs1 and 2, then iterate to 22 
	fullchr = chrs[0].union_rows(chrs[1])
	for i in range(2,len(chrs)):
		print("adding chromosome " + str(numbers[i]))
		fullchr = fullchr.union_rows(chrs[i])

	return fullchr 

def annotate(fullchr, numbers):
	""" import phenotype & convert to table object, remove NaNs, join to matrixTable """
	print("importing phenotype data...")
	phenos = pd.read_pickle("./phenos.pkl")
	phenos.columns = ["eid","BMI"]
	ph = hl.Table.from_pandas(phenos,key=["eid"])
	ph_filtered = ph.filter(hl.is_nan(ph.BMI),keep=False)
	#pprint(ph.count()-ph_filtered.count())
	fullchr = fullchr.annotate_cols(phenotype = ph_filtered[fullchr.s])
	# get biological sex and ethnicity from uk biobank data, remove NaNs, and add it to table 
	print("importing additional data...")
	bios = pd.read_csv("ukb37493(2).csv", usecols = ["eid","22001-0.0","22006-0.0"], dtype={"eid":str,"22001-0.0":float,"22006-0.0":float})
	bios.columns = ["eid","sex","ethnic_group"]
	bs = hl.Table.from_pandas(bios,key=["eid"])
	bs = bs.filter(hl.is_nan(bs.sex),keep=False)
	bs = bs.filter(hl.is_nan(bs.ethnic_group),keep=False)
	fullchr = fullchr.annotate_cols(ukbb = bs[fullchr.s])
	# import info score and MAF for specified chromosomes, convert to tables, join to each other 
	scores = []
	for i in numbers:
		print("getting MAF and info for chromosome " + str(i) + "...")
		filename = "ukb_mfi_chr"+str(i)+"_v3.txt"
		infos = pd.read_csv(filename, sep="\t", header=None)
		ins = infos[[0,5,6,7]]
		ins.columns = ["varid","MAF","minor_allele","info_score"]
		i = hl.Table.from_pandas(ins,key=["varid"])
		scores.append(i)
	if len(numbers) > 1:
		print("joining scores for all chromosomes...")
		fullscr = scores[0].union(scores[1])
		for i in range(2,len(scores)):
			fullscr = fullscr.union(scores[i])
	else:
		fullscr = scores[0]
	fullchr = fullchr.annotate_rows(SNP_info = fullscr[fullchr.varid])
	# import reported sex and YOB, conver"t YOB to age, annotate matrixTable
	print("getting reported age and sex...")
	p = pd.read_csv("ukb40744.csv", usecols = ["eid","31-0.0","34-0.0"], dtype={"eid":str,"31-0.0":float,"34-0.0":float})
	p.columns = ["eid","reported_sex","age"]
	p.age = int(d.datetime.now().year) - p.age
	pt = hl.Table.from_pandas(p, key=["eid"])
	fullchr = fullchr.annotate_cols(ukbb2 = pt[fullchr.s])

	return fullchr


######   SAMPLE QC   ######

def filter_samples(fullchr):
	""" exclude recommended samples from analysis"""
	# exclude non-european individuals 
	print('removing other ethnicities...')
	fullchr_qc = fullchr.filter_cols(fullchr.ukbb.ethnic_group == 1)

	# exclude individuals not used in Biobank PCA calculation
	print("removing non-PCA samples...")
	p = pd.read_csv("ukb37493(2).csv", usecols = ["eid","22020-0.0"], dtype={"eid":str,"22020-0.0":float})
	p.columns = ["eid","pca_inclusions"]
	pca_inclusions = hl.Table.from_pandas(p,key=["eid"])
	pca_inclusions = pca_inclusions.filter(hl.is_nan(pca_inclusions.pca_inclusions),keep=False)
	to_keep = pca_inclusions.eid.collect()
	set_to_keep = hl.literal(to_keep)
	#print('%d individuals used in PCA calc' % hl.eval(hl.len(set_to_keep)))
	fullchr_qc = fullchr_qc.filter_cols(set_to_keep.contains(fullchr_qc['s']))
	#print('non-PCA samples removed: %d' % hl.eval(fullchr_qc1.count_cols()-fullchr_qc2.count_cols()))

	# exclude biological and reported sex mismatches
	print("removing sex mismatches...")
	fullchr_qc = fullchr_qc.filter_cols(fullchr_qc.ukbb.sex == fullchr_qc.ukbb2.reported_sex)

	# import lists of recommended exclusions and weird cases, convert to tables, filter for NaNs 

	# recommended heterozygosity exclusions
	he = pd.read_csv("ukb37493(2).csv", 
	                   usecols = ["eid","22010-0.0"],
	                   dtype={"eid":str,"22010-0.0":float})
	he.columns = ["eid","het_exclusions"]
	het_exclusions = hl.Table.from_pandas(he,key=["eid"])
	het_exclusions = het_exclusions.filter(hl.is_nan(het_exclusions.het_exclusions),keep=False)

	# recommended relatedness exclusions
	re = pd.read_csv("ukb37493(2).csv", 
	                   usecols = ["eid","22018-0.0"],
	                   dtype={"eid":str,"22018-0.0":float})
	re.columns = ["eid","rel_exclusions"]
	rel_exclusions = hl.Table.from_pandas(re,key=["eid"])
	rel_exclusions = rel_exclusions.filter(hl.is_nan(rel_exclusions.rel_exclusions),keep=False)

	# aneuploidy cases 
	an = pd.read_csv("ukb37493(2).csv", 
	                   usecols = ["eid","22019-0.0"],
	                   dtype={"eid":str,"22019-0.0":float})
	an.columns = ["eid","aneuploidy"]
	aneuploidy = hl.Table.from_pandas(an,key=["eid"])
	aneuploidy = aneuploidy.filter(hl.is_nan(aneuploidy.aneuploidy),keep=False)

	# remove samples from table if they appear in recommended exclusion lists 

	# heterozygosity exclusions
	to_remove = het_exclusions.eid.collect()
	set_to_remove = hl.literal(to_remove)
	#print('to remove: %d' % hl.eval(hl.len(set_to_remove)))
	print("removing heterozygosity outliers...")
	fullchr_qc = fullchr_qc.filter_cols(~set_to_remove.contains(fullchr_qc['s']))
	#print('Het outlier samples removed: %d' % hl.eval(fullchr_qc2.count_cols()-fullchr_qc3.count_cols()))

	# relatedness exclusions
	to_remove = rel_exclusions.eid.collect()
	set_to_remove = hl.literal(to_remove)
	#print('to remove: %d' % hl.eval(hl.len(set_to_remove)))
	print("removing recommended relatedness exclusions...")
	fullchr_qc = fullchr_qc.filter_cols(~set_to_remove.contains(fullchr_qc['s']))
	#print('Relatedness exclusion samples removed: %d' % hl.eval(fullchr_qc3.count_cols()-fullchr_qc4.count_cols()))

	# aneuploidy
	to_remove = aneuploidy.eid.collect()
	set_to_remove = hl.literal(to_remove)
	#print('to remove: %d' % hl.eval(hl.len(set_to_remove)))
	print("removing aneuploidy cases...")
	fullchr_qc = fullchr_qc.filter_cols(~set_to_remove.contains(fullchr_qc['s']))
	#print('Aneuploidy samples removed: %d' % hl.eval(fullchr_qc4.count_cols()-fullchr_qc5.count_cols()))

	print('Total samples remaining: %d' % hl.eval(fullchr_qc.count_cols()))

	return fullchr_qc

######   VARIANT QC   ###### 

def filter_variants(fullchr_qc):
	""" exclude recommended variants from analysis """
	# calculate per-variant qc stats
	fullchr_qc2 = hl.variant_qc(fullchr_qc)
	#print('Total Variants: %d' % (fullchr_qc2.count_rows()))

	# remove non-SNPs
	print("removing non SNPs...")
	fullchr_qc2 = hl.filter_alleles(fullchr_qc2, lambda allele, i: hl.is_snp(fullchr_qc2.alleles[0], allele))
	#print('Variants remaining after non-SNP exclusion: %d' % (fullchr_qc2.count_rows()))

	# HWE filter (Neale lab value)
	print("removing SNPs outside hardy-weinberg equilibrium...")
	fullchr_qc2 = fullchr_qc2.filter_rows(fullchr_qc2.variant_qc.p_value_hwe > 1e-10)
	#print('Variants remaining after HWE filter: %d' % (fullchr_qc2.count_rows()))

	# info score filter -- using 0.4 as indicated by exploratory plot
	print(">>>>>>UPDATED removing SNPs with info score less than 0.8...")
	fullchr_qc2 = fullchr_qc2.filter_rows(fullchr_qc2.SNP_info.info_score > 0.8)
	#print('Variants remaining after info score filter: %d' % (fullchr_qc2.count_rows()))

	# MAF filter
	print("removing SNps with minor allele frequency less than 0.001...")
	fullchr_qc2 = fullchr_qc2.filter_rows(fullchr_qc2.SNP_info.MAF > 0.001)
	#print('Variants remaining after MAF filter: %d' % (fullchr_qc2.count_rows()))

	# print("removing SNPs with low call rate...")
	# fullchr_qc2 = fullchr_qc2.filter_rows(fullchr_qc2.variant_qc.call_rate > 0.95)

	#print("removing multiallelic variants...")
	#fullchr_qc2 = fullchr_qc2.filter_rows()

	#print("Variants remaining after variant filter: %d" % (fullchr_qc2.count_rows()))

	return fullchr_qc2

#####   GWAS   #####

def get_PCA(fullchr_qc2):
	# import principal components calculated by uk biobank & join to matrixtable
	p = pd.read_csv("ukb37493(2).csv", 
	               usecols = ["eid","22009-0.1","22009-0.2","22009-0.3",
	                          "22009-0.4","22009-0.5","22009-0.6",
	                          "22009-0.7","22009-0.8","22009-0.9","22009-0.10"],
	               dtype={"eid":str,"22009-0.1":float,"22009-0.2":float,"22009-0.3":float,
	                          "22009-0.4":float,"22009-0.5":float,"22009-0.6":float,
	                          "22009-0.7":float,"22009-0.8":float,"22009-0.9":float,"22009-0.10":float})
	p.columns = ["eid","pca1","pca2","pca3","pca4","pca5","pca6","pca7","pca8","pca9","pca10"]
	pca = hl.Table.from_pandas(p,key=["eid"])
	fullchr_qc2 = fullchr_qc2.annotate_cols(PCA = pca[fullchr_qc2.s])

	return fullchr_qc2

def gwas(fullchr_qc2, cfdrs, dosage):
	if dosage:
		data = fullchr_qc2.dosage
	else:
		data = fullchr_qc2.GT.n_alt_alleles()
	if cfdrs:
		# GWAS with dosages and confounders as covariates 
		print("running linear regression with confounders...")
		gwas = hl.linear_regression_rows(y=fullchr_qc2.phenotype.BMI, x=data,
		                                 covariates=[1.0, fullchr_qc2.ukbb2.reported_sex, 
		                                             fullchr_qc2.ukbb2.age,
		                                             fullchr_qc2.PCA.pca1, fullchr_qc2.PCA.pca2, 
		                                             fullchr_qc2.PCA.pca3,fullchr_qc2.PCA.pca4, 
		                                             fullchr_qc2.PCA.pca5, fullchr_qc2.PCA.pca6,
		                                             fullchr_qc2.PCA.pca7, fullchr_qc2.PCA.pca8, 
		                                             fullchr_qc2.PCA.pca9,fullchr_qc2.PCA.pca10])
	else:
			# GWAS with dosages but without confounders as covariates 
		print("running linear regression without confounders...")
		gwas = hl.linear_regression_rows(y=fullchr_qc2.phenotype.BMI, x=data, covariates=[1.0])

	gwas_filtered = gwas.filter(hl.is_nan(gwas.p_value),keep=False)
	return gwas_filtered

def plot(gwas_filtered, cfdrs, dosage, output_string):
	if dosage: 
		data = "_dosage_"
	else:
		data = "_GT_"
	if cfdrs:
		output_strings = [output_string+data+"_qq_covariates.html",output_string+data+"_mhtn_covariates.html"]
	else:
		output_strings = [output_string+data+"_qq_nocovariates.html",output_string+data+"_mhtn_nocovariates.html"]

	output_file(output_strings[0])
	p = hl.plot.qq(gwas_filtered.p_value)
	save(p)
	output_file(output_strings[1])
	p = hl.plot.manhattan(gwas_filtered.p_value)
	save(p)









