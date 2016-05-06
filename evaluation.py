import numpy as np
import pandas
import math
import os


os.chdir('/Users/samhita/Documents/0Princeton/1_SophomoreYear/COS435/FinalProject/Results/Places/Yard/')

test = pandas.read_csv('503627556_0e3b3b1498_w3_results.csv', header=None)

pool_size = 35
pool_relDocs = 22


# precision
counter = 0
for i in range(0, len(test)):  #change end of for loop for precision @rank
	if test.iloc[i, 0] != 0:
		counter = counter + 1
precision = (float)(counter)/len(test)
print 'Precision: ', precision


# recall
counter = 0
for i in range(0, len(test)):
	if test.iloc[i, 0] != 0:
		counter = counter + 1
recall = (float)(counter)/pool_relDocs
print 'Recall: ', recall


# reciprocal rank
rank = 0;
for i in range(0, len(test)):
	if test.iloc[i, 0] != 0:
		rank = i+1
		break;
if rank == 0:
	recip_rank = 0;
else:
	recip_rank = 1/(float)(rank)
print 'Reciprocal rank: ', recip_rank


# discounted cumulative grain (DCG)
DCG = 0;
for i in range(0, len(test)):
	temp = (float)(test.iloc[i, 0])/math.log(i+2, 2)
	DCG = DCG + temp
print 'DCG: ', DCG


# expected reciprocal rank (ERR)
def rscore(max_score, score):
	return ((float)(1)/math.pow(2,max_score))*(math.pow(2, score) - 1)
ERR = 0
for j in range(0, len(test)):
	product = 1
	for k in range (0, j-1):
		product = product*(1-rscore(5, test.iloc[k, 0]))
	ERR = ERR + (float)(1)/(j+1) * product * rscore(5, test.iloc[j, 0])
print 'ERR: ', ERR
