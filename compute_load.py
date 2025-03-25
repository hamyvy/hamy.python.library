import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument( "--nchr",)	
parser.add_argument( "--vartype",)	
args = parser.parse_args()
chrom = args.nchr
vartype = args.vartype

DATA_DIR = ''

#-------------------------------------------------------------------------------------------------
rsids = []
rsids.append(['0','0'])
mapsnp = open(DATA_DIR+'genotype/var_filtered/rsid_'+str(chrom)+'.txt','r')
for line in mapsnp:
	rsids.append(line.strip().split(' ')[1])
print(len(rsids))

#-------------------------------------------------------------------------------------------------

inf1 = open(DATA_DIR+'LoF/LOF_HC_allchr.txt','r')

hd = inf1.readline().strip().split('\t')
colneed = ["CHROM","ID","Gene","f","PHRED","caddratio","shet","HWE","UKBQC","vartype"]
colindex = [hd.index(colname) for colname in colneed]
print(colindex)

snpinfo, Shet = {}, {}
for line in inf1:
	line = line.replace('NA','0')
	brl = line.strip().split("\t")
	if brl[colindex[0]] == chrom:
		if brl[colindex[9]] == vartype:
			if brl[colindex[7]] == '1' and brl[colindex[8]]=='1':
				snpinfo[brl[colindex[1]]] = [brl[colindex[2]],float(brl[colindex[3]]),float(brl[colindex[4]]),float(brl[colindex[5]]),float(brl[colindex[6]])]
				if brl[colindex[2]] not in Shet.keys():
					Shet[brl[colindex[2]]] = float(brl[colindex[6]])
				else: 
					if float(brl[colindex[6]])>Shet[brl[colindex[2]]]: 
						Shet[brl[colindex[2]]] = float(brl[colindex[6]])
inf1.close()

print(len(snpinfo))

#-------------------------------------------------------------------------------------------------

def _do_sum(rsids, maf, weight):
	Gscore = [0,0,0,0] # LOF count, sum(LOF*CADD), sum(LOF*CADDratio*s_het), sum(s_het)
	gene_list = set()
	for snp in rsids:
		try:
			if snpinfo[snp][1]<maf:
				Gscore[0] += weight
				Gscore[1] += snpinfo[snp][2]
				Gscore[2] += snpinfo[snp][3]*snpinfo[snp][4]
				gene_list.add(snpinfo[snp][0])
		except KeyError:
			pass
	Gscore[3] =sum([Shet[gene] for gene in gene_list])*weight
	Gscore[1] = Gscore[1]*weight
	Gscore[2] = Gscore[2]*weight
	return np.array(Gscore)
#-------------------------------------------------------------------------------------------------
def GeneGrouping(rsids, maf, weight):
	newrsids = [i for i in rsids if i in snpinfo.keys() and snpinfo[i][1]<maf]
	gene_list = [snpinfo[snp][0] for snp in newrsids]
	genecount = [gene_list.count(key)*weight for key in Shet.keys()]
	return np.array(genecount)
#-------------------------------------------------------------------------------------------------
def GeneGrouping_weightCADD(rsids, weight):
	newrsids = [i for i in rsids if i in snpinfo.keys()]
	gene_list = [snpinfo[snp][0] for snp in newrsids]
	Nsnpweighted = {}
	for g in set(gene_list): Nsnpweighted[g] = 0
	for snp in newrsids: Nsnpweighted[snpinfo[snp][0]] += snpinfo[snp][3]*weight
	genecount2 = []
	for key in Shet.keys():
		try:
			genecount2.append(Nsnpweighted[key])
		except KeyError:
			genecount2.append(0)
	return np.array(genecount2)
#-------------------------------------------------------------------------------------------------
AFthreshold = [0.000005,0.00001,0.00005,0.0001,0.0005,0.001,0.01,0.1,0.5]
AFabb = ["f5e6","f1e5","f5e5","f1e4","f5e4","f1e3","f1e2","f1e1","f5e1"]

outf = open("per_chrom/"+vartype+"_c"+str(chrom)+".txt","w")
outf.write('ID')
for i in range(len(AFthreshold)): 
	for j in range(4):
		outf.write(" "+AFabb[i]+"_s"+str(j))
outf.write("\n")

outf2 = open("nlof/"+vartype+"_c"+str(chrom)+".txt","w")
outf2.write('ID '+' '.join([key for key in Shet.keys()])+'\n')
outf3 = open("weightCADD/"+vartype+"_c"+str(chrom)+".txt","w")
outf3.write('ID '+' '.join([key for key in Shet.keys()])+'\n')

inf = open(DATA_DIR+"genotype/var_filtered/snplist_"+chrom+".txt","r")
inf.readline()
for line in inf:
	brl = line.split()

	if len(brl)==2:
		hets = [rsids[int(i)] for i in brl[1].split(',')]
		outf.write(brl[0])
		for maf in AFthreshold:
			outf.write(' ' + ' '.join(str(item) for item in _do_sum(hets,maf,1)) )
		outf.write('\n')
		outf2.write(brl[0]+' '+ ' '.join(str(item) for item in GeneGrouping(hets,0.00001,1))+'\n')
		outf3.write(brl[0]+' '+ ' '.join(str(item) for item in GeneGrouping_weightCADD(hets,1))+'\n')
	elif len(brl)==3:
		hets = [rsids[int(i)] for i in brl[1].split(',')]
		homs = [rsids[int(i)] for i in brl[2].split(',')]
		outf.write(brl[0])
		for maf in AFthreshold:
			outf.write(' '+ ' '.join(str(item) for item in _do_sum(hets,maf,1) + _do_sum(homs,maf,2)) )
		outf.write("\n")
		outf2.write(brl[0]+' '+ ' '.join(str(item) for item in GeneGrouping(hets,0.00001,1) + GeneGrouping(homs,0.00001,2))+'\n')
		outf3.write(brl[0]+' '+ ' '.join(str(item) for item in GeneGrouping_weightCADD(hets,1) + GeneGrouping_weightCADD(homs,2))+'\n')

inf.close()
