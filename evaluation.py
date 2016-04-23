import numpy as np
import pandas
import math
import os


os.chdir('/Users/samhita/Documents/0Princeton/1_SophomoreYear/COS435/FinalProject')

test = pandas.read_csv('test.csv', header=None)

print test

pool_size = 35
pool_relDocs = 14


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
for i in range(0, len(test)):
	if test.iloc[i, 0] != 0:
		rank = i+1
		break;
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
