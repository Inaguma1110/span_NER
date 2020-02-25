import argparse
import re
import os
import configparser
import shelve
import itertools
import sys, codecs
import collections
import six
import pdb
import tqdm

import numpy as np
import torch
import sentencepiece as spm
import tensorflow as tf
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")
    
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token, _ = token.split("\t")
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def main(config):
    data_path = config.get('path', 'DATA_PATH')
    max_length = int(config.get('makedata', 'MAX_SENT_LEN'))
    
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('structure',help='NEST_INNER -> All nest entity\nNEST_OUTER -> only out nest entity')
    data_structure_type = parser.parse_args().structure

    try:
        if data_structure_type == 'NEST_INNER' or data_structure_type == 'NEST_OUTER':
            print('data structure is {}'.format(data_structure_type))
        else:
            raise ValueError('CHOOSE DATA STRUCTURE ! (NEST_INNER or NEST_OUTER)')
    except ValueError as e:
        print('\n{}'.format(e))
        sys.exit()

    filelst = os.listdir(data_path)
    doclist = []
    sp = spm.SentencePieceProcessor()
    sp.Load('./spm_model/wiki-ja.model')
    tokenizer = BertTokenizer.from_pretrained('./spm_model/wiki-ja.vocab.txt')
    Rel_None_Label = config.get('makedata', 'REL_NONE_LABEL')
    REL_DIC = dict()  #全体の関係辞書
    REL_DIC[Rel_None_Label] = 0
    REL_LABEL_DICT = {}
    relation_mapping = {'NegativeLinear':'Negative', 'Linear':'Relation', 'Square':'Relation'}

    rel_index = 1
    Num_of_entity = 0
    Num_of_rm_entity = 0
    Num_of_data = 0
    Num_of_miss_tokenize = 0
    miss_tokenize_list = [0]
    hitotsu = 0
    kuttuki = 0
    corpus = []

    filename_lst = []

    # dataname = 'Machining_data_kakogijutsu_span_NER_RE_pretrain_Japanese_BERT_maxlength_100'
    dataname = config.get('dataname', 'REL')
    print('\n save dataname is {0}\n'.format(dataname))
    only_txt_lst = []
    for txtfile in filelst:
        docpair = ()
        if (txtfile[-4:]) == '.txt':
            for annfile in filelst:
                if (annfile[-4:]) == '.ann' and annfile[:-4] == txtfile[:-4]:
                    docpair = (txtfile, annfile)
                    doclist.append(docpair)
            if len(docpair) == 0:
                only_txt_lst.append(txtfile)

    # pdb.set_trace()

    for n, doc in enumerate(tqdm.tqdm(doclist)):
        # pdb.set_trace()
        # print('create data of ' + str(doc[0]))
        filename_lst.append(doc[0][:-4])
        Entdic = dict()
        EntityList = []
        removed_Entdic = dict()
        removed_EntityList = []
        Reldic = dict()
        REL_LABEL_unit = []
        Triggerdic = dict()
        vocab = load_vocab('./spm_model/wiki-ja.vocab')
        
        # doc = ('endmil_0029_558.txt','endmil_0029_558.ann')        
        max_length = int(config.get('makedata', 'MAX_SENT_LEN'))

        with open(data_path + doc[0], 'r', encoding='utf-8') as txtfile:
            text = txtfile.read()

        with open(data_path + doc[1], 'r', encoding='utf-8') as annfile:
            annotations = annfile.read().split('\n')
        
        #pdb.set_trace()
        ########################################## ann ファイル内を処理．辞書作成 ###############################################
        for line in annotations:
            instance = line.split('\t')
            if 'T' in instance[0]: #instance=['T0', 'Machining 4 9', '上向き削り']
                instance_tag = instance[1].split(' ')[0] #instance_tag = 'Machining'
                if 'Trigger' not in instance_tag: #TriggerタグではなかったらEntitiy
                    Entdic[instance[0]] = (instance[1].split()[0],int(instance[1].split()[1]),int(instance[1].split()[2]),instance[2])
                    EntityList.append(instance[0])
                    #Entdic[T0] = (Machihing, 4, 9, '上向き削り') 
                else:
                    Triggerdic[instance[0]] = (instance[1].split()[0],instance[1].split()[1],instance[1].split()[2],instance[2])

            elif 'R' in instance[0]: #instance = ['R1', 'Relation Arg1:T30 Arg2:T25', '']
                instance_tag = instance[1].split(' ')[0]
                Reldic[instance[0]] = (instance[1].split(' ')[0],instance[1].split(' ')[1][5:],instance[1].split(' ')[2][5:])

                if instance_tag in relation_mapping:
                    instance_tag = relation_mapping[instance_tag]

                if instance_tag not in REL_DIC:
                    REL_DIC[instance_tag] = rel_index
                    rel_index += 1
        Num_of_entity += len(Entdic)
        #################################################################################################################


		##################  Remove NEST entity pair and create for example 切削 and 切削抵抗  ######################### 
        for key1, val1 in Entdic.items(): # key1が省かれるかどうか調べる
            sta1,end1 = Entdic[key1][1],Entdic[key1][2]
            flag = True
            for key2 in Entdic.keys():
                sta2,end2 = Entdic[key2][1],Entdic[key2][2]
                if sta1 == sta2 and end1 < end2:     #　切削　と　切削抵抗　の場合
                    flag = False
                if sta1 > sta2 and end1 == end2:     #　抵抗　と　切削抵抗　の場合
                    flag = False
                if sta2 < sta1 and end1 < end2:      #　抵抗　と　切削抵抗力　の場合
                    flag = False
                if sta1 == sta2 and end1 == end2:
                    continue
            if flag == True:
                removed_Entdic[key1] = val1
                removed_EntityList.append(key1)
        Num_of_rm_entity += len(removed_Entdic)
        #################################################################################################################


        # main prrocess ##################################################################################################################################################################

        # set dictionary ###############################################################################################
        try:
            if data_structure_type == 'NEST_INNER':
                TARGET_DIC = Entdic
                final_num_of_entity = Num_of_entity 
            elif data_structure_type == 'NEST_OUTER':
                TARGET_DIC = removed_Entdic
                final_num_of_entity = Num_of_rm_entity
            else:
                raise ValueError('CHOOSE DATA STRUCTURE ! (NEST_INNER or NEST_OUTER)')
        except ValueError as e:
            print('\n{}'.format(e))
            sys.exit()

        #################################################################################################################


        # pdb.set_trace()
        spmed = sp.EncodeAsPieces(text)


        if len(spmed[0]) == 1:
            spmed[0] = '[CLS]'
            hitotsu += 1
        else:
            spmed[0] = spmed[0][1:]
            spmed.insert(0, '[CLS]')
            kuttuki += 1
        spmed.append('[SEP]')


        #Paddingは0で置き換え
        output_film_size1   = [0] * (max_length - 1)
        output_film_size2   = [0] * (max_length - 1)
        output_film_size3   = [0] * (max_length - 1)
        output_film_size4   = [0] * (max_length - 1)

        tag_dict = {} # tag_dict['T1631'] = (outfilmのサイズ, 何トークン目か)
        

        attention_mask      = [1] * len(spmed)

        target_spm          = [] 
        start_cnt = 0
        end_cnt   = 0

        for x, subword in enumerate(spmed[1:-1]):
            end_cnt = len(subword)
            target_spm.append((x, subword, start_cnt, start_cnt + end_cnt))
            start_cnt += end_cnt
        
        for x1 in range(len(target_spm)):
            x2 = x1 + 1
            x3 = x1 + 2
            x4 = x1 + 3

            # subwordが存在するindexは0に置き換える
            output_film_size1[x1] = 0
            output_film_size2[x1] = 0
            output_film_size3[x1] = 0
            output_film_size4[x1] = 0


            for k, v in TARGET_DIC.items():      #Entityの存在する部分にfilter毎に1を立てる
                try:

                    if target_spm[x1][2] == v[1] and target_spm[x1][3] == v[2]:
                        output_film_size1[x1] = 1
                        tag_dict[k] = (1,target_spm[x1])

                    if target_spm[x1][2] == v[1] and target_spm[x2][3] == v[2]:
                        output_film_size2[x1] = 1
                        tag_dict[k] = (2,target_spm[x1])

                    if target_spm[x1][2] == v[1] and target_spm[x3][3] == v[2]:
                        output_film_size3[x1] = 1
                        tag_dict[k] = (3,target_spm[x1])

                    if target_spm[x1][2] == v[1] and target_spm[x4][3] == v[2]:
                        output_film_size4[x1] = 1
                        tag_dict[k] = (4,target_spm[x1])

                except IndexError:
                    pass

        # pdb.set_trace()
        for miss_unit in removed_EntityList:
            try:
                tag_dict[miss_unit]
            except:
                Num_of_miss_tokenize += 1
                miss_tokenize_list.append((doc[0],Entdic[miss_unit]))
        

        #[CLS]の分insert
        output_film_size1.insert(0,0)
        output_film_size2.insert(0,0)
        output_film_size3.insert(0,0)
        output_film_size4.insert(0,0)
        # attention_mask[0] = 1

        for i in range(max_length - len(spmed)):
            spmed.append('[PAD]')
            attention_mask.append(0)
        # pdb.set_trace()

        indx_tokens = [vocab[s] if s in vocab else vocab['<unk>'] for s in spmed]


        for rel_v in Reldic.values():
            try:
                REL_LABEL_unit.append((REL_DIC[rel_v[0]],(tag_dict[rel_v[1]],tag_dict[rel_v[2]])))
            except:
                pass
        REL_LABEL_DICT[n] = REL_LABEL_unit

        corpus.append(((n,doc,Entdic,Reldic), indx_tokens, output_film_size1, output_film_size2, output_film_size3, output_film_size4, attention_mask, spmed))
        Num_of_data += 1

    #         pdb.set_trace()
    #     pdb.set_trace()
    pdb.set_trace()
    print('_ only separated num is {0}'.format(hitotsu))
    print('_*** not separated num is {0}'.format(kuttuki))
    print('Number of entity including NEST structure {0}'.format(final_num_of_entity))
    print('Number of relation ' + str(Num_of_data))

    print('Number of Miss tokenize is {}'.format(Num_of_miss_tokenize))
    # print(miss_tokenize_list)
    database = shelve.open(config.get('path', 'SHELVE_PATH'))
    database[dataname] = [vocab, REL_DIC, corpus, filename_lst,REL_LABEL_DICT]

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('../machine.conf')
    main(config)
