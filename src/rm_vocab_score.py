import pdb
import os

inputpath = './spm_model/'

outputpath = './spm_model/'


vocabfile = open(inputpath + 'wiki-ja.vocab', "r")
txtfile   = open(outputpath + 'wiki-ja.vocab.txt', "a")
for token in vocabfile:
    a = token.split('\t')[0]
    txtfile.write(a + '\n')
