import gzip
import sys
import numpy as np

infilename = sys.argv[1]
outfilename = sys.argv[2]

outf = gzip.open(outfilename, 'wt')
inf = gzip.open(infilename,'rt')

refcode = ["0","."]
nheader = 0

for line in inf: 
	#line=line.decode()
	
	if line[0] == "c":
		brl = line.split("\t")

		#if the site is multiallelic:
		if ";" in brl[2]: 
			alts = brl[4].split(",")
			Nalt = len(alts)

			ids = brl[2].split(";")
			DPs,poss,altrank = [],[],[]
			#set DP threshold for SNP(DP=7) and indel(DP=10)
			for snp in range(Nalt):
				brkID = ids[snp].split("_")
				poss.append(int(brkID[1]))
				if len(brkID[2])==1 and len(brkID[3])==1: DPs.append(7)
				else: DPs.append(10)
			#to order snps according to position on the genome	
			altrank = np.argsort(poss)

			infos = brl[7].split(";")
			AFs = infos[0].split("=")[1].split(",")
			AQs = infos[1].split("=")[1].split(",")

			#obtain new array of genotypes for each single snp
			genolist = [] 
			for ii in range(9,len(brl)):
				genoinfo = brl[ii].split(":")

				if genoinfo[2]==".": AD = ['.' for xx in range(Nalt+1)]
				else: AD = genoinfo[2].split(",")

				if genoinfo[4]==".": PL = ['.' for xx in range( (Nalt*(Nalt+1)/2) + Nalt+1)]
				else: PL = genoinfo[4].split(",") 
				
				genoj = []
				for jj in range(1,Nalt+1):
					if int(genoinfo[1]) >= DPs[jj-1]:
					#if the genotype pass DP thhreshold
						if genoinfo[0][0] in refcode: A1=genoinfo[0][0]
						elif genoinfo[0][0] == str(jj): A1="1"
						else: A1="."
						if genoinfo[0][2] in refcode: A2=genoinfo[0][2]
						elif genoinfo[0][2] == str(jj): A2="1"
						else: A2="."

						newPL = PL[0]+","+PL[(jj*(jj+1)//2)]+","+PL[((jj*(jj+1)//2)+jj)] #PL order: F(j/k) = (k*(k+1)/2)+j
						
						genoj.append(A1+"/"+A2+":"+genoinfo[1]+":"+AD[0]+","+AD[jj]+":"+genoinfo[3]+":"+newPL+":"+genoinfo[5])

					else: genoj.append("./.:"+genoinfo[1]+":"+AD[0]+","+AD[jj]+":"+genoinfo[3]+":0,0,0:"+genoinfo[5])
				genolist.append(genoj)

			#write output
			for snp in altrank: 
				brkID = ids[snp].split("_")
				outf.write((brl[0]+"\t"+brkID[1]+"\t"+ids[snp]+"\t"+brkID[2]+"\t"+brkID[3]+"\t"+brl[5]+ \
					"\t"+brl[6]+"	AF="+AFs[snp]+";AQ="+AQs[snp]+"\t"+brl[8]))
				for gn in genolist: outf.write(("\t"+gn[snp]))

		#if the site is biallelic:
		else:
			if len(brl[3])==1 and len(brl[4])==1: DP=7
			else: DP=10
			outf.write("\t".join(brl[:9]))
			for ii in range(9,len(brl)):
				if int(brl[ii].split(":")[1]) >= DP: outf.write(("\t"+brl[ii]))
				else: outf.write(("	./."+brl[ii][3:]))

	elif line[0:2] == "##":
		#only keep 42 header lines
		if nheader<42: outf.write(line)
		nheader += 1
		continue
	elif line[0] == "#": 
		outf.write(line)
		continue
outf.close()
inf.close()


