import sys
import MeCab as mcb
import sentencepiece as spm
import re
import os
import configparser
import pdb
import shelve
import numpy as np
import itertools
import sys, codecs

def main(config):
    sp = spm.SentencePieceProcessor()
    sp.Load("../published_model/bert_spm/wiki-ja.model")
    data_path = config.get('path', 'DATA_PATH')
    filelst = os.listdir(data_path)
    doclist = []
    WORD_DIC = dict() #全体の単語辞書
    REL_DIC = dict()  #全体の関係辞書
    REL_DIC['None'] = 0
    word_index = 1
    rel_index = 1
    Num_of_data = 0

    corpus = []

    filename_lst = []

    new_filelst = []

    dataname = 'Machining_data_kakogijutsu_span_NER_RE'

    # if dataname == 'Machining_data_kakogijutsu': # seimitsuとendmillファイルは対象外とする．
    #     for fname in filelst:
    #         if "seimitu" in fname or "endmil" in fname:
    #             continue
    #         new_filelst.append(fname)    
    # else:
    #     new_filelst = filelst


    for txtfile in new_filelst:
        docpair = ()
        if (txtfile[-4:]) == '.txt':
            for annfile in new_filelst:
                if (annfile[-4:]) == '.ann' and annfile[:-4] == txtfile[:-4]:
                    docpair = (txtfile, annfile)
                    doclist.append(docpair)

    #pdb.set_trace()

    for n, doc in enumerate(doclist):
        print('create data of ' + str(doc[0]))
        filename_lst.append(doc[0][:-4])
        #pdb.set_trace()
        Entdic = dict()
        Reldic = dict()
        Triggerdic = dict()

        
        with open(data_path + doc[0], 'r', encoding='utf-8') as txtfile:
            text = txtfile.read()

        with open(data_path + doc[1], 'r', encoding='utf-8') as annfile:
            annotations = annfile.read().split('\n')
        
        #pdb.set_trace()
        for line in annotations:
            instance = line.split('\t')
            if 'T' in instance[0]: #instance=['T0', 'Machining 4 9', '上向き削り']
                instance_tag = instance[1].split(' ')[0] #instance_tag = 'Machining'
                if 'Trigger' not in instance_tag: #TriggerタグではなかったらEntitiy
                    Entdic[instance[0]] = (instance[1].split()[0],int(instance[1].split()[1]),int(instance[1].split()[2]),instance[2])
                    #Entdic[T0] = (Machihing, 4, 9, '上向き削り') 
                else:
                    Triggerdic[instance[0]] = (instance[1].split()[0],instance[1].split()[1],instance[1].split()[2],instance[2])

            elif 'R' in instance[0]: #instance = ['R1', 'Relation Arg1:T30 Arg2:T25', '']
                instance_tag = instance[1].split(' ')[0]
                Reldic[instance[0]] = (instance[1].split(' ')[0],instance[1].split(' ')[1][5:],instance[1].split(' ')[2][5:])
                if instance_tag not in REL_DIC:
                    REL_DIC[instance_tag] = rel_index
                    rel_index += 1

	
        spmed = sp.EncodeAsPieces(text)
#        ps = 0
#        pe = 0
#        for i in spmed[1:]:
#            pe = len(i)
#            print(i,ps,ps+pe)
#            ps += pe


#        ent_position_map    = [0] * 150
#        other_position_map  = [0] * 150
#        firstpositionmap    = [0] * 150
#        secondpositionmap   = [0] * 150
        output_film_size1   = [0] * 150
        output_film_size2   = [0] * 150
        output_film_size3   = [0] * 150
        output_film_size4   = [0] * 150

        target_spm          = [] 
        #pdb.set_trace()
        start_cnt = 0
        end_cnt   = 0

        for x, subword in enumerate(spmed[1:]):
            if subword not in WORD_DIC:
                WORD_DIC[subword] = word_index
                word_index += 1
            sentence_word_map[x] = WORD_DIC[subword]

            end_cnt = len(subword)
            target_spm.append((x,subword,start_cnt, start_cnt + end_cnt))
            start_cnt += end_cnt

        for x1 in range(len(target_spm)):
            x2 = x1 + 1
            x3 = x1 + 2
            x4 = x1 + 3
            for k, v in Entdic.items():


                if target_spm[x1][2] == v[1] and target_spm[x1][3] == v[2]:
                    output_film_size1[x1] = 1

                
                try:

                    if target_spm[x1][2] == v[1] and target_spm[x2][3] == v[2]:
                        output_film_size2[x1] = 1


                    if target_spm[x1][2] == v[1] and target_spm[x3][3] == v[2]:
                        output_film_size3[x1] = 1


                    if target_spm[x1][2] == v[1] and target_spm[x4][3] == v[2]:
                        output_film_size4[x1] = 1


                except IndexError:
                    pass

        corpus.append((doc[0], sentence_word_map, output_film_size1, output_film_size2, output_film_size3, output_film_size4, (n,doc,Entdic, Reldic)))

        Num_of_data += 1
        print()
        #pdb.set_trace()
    pdb.set_trace()
    print('Number of data ' + str(Num_of_data))
    database = shelve.open(config.get('path', 'SHELVE_PATH'))
    database[dataname] = [WORD_DIC, REL_DIC, corpus, filename_lst]

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('../machine.conf')
    main(config)
