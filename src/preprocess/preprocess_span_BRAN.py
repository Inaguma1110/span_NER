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
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig


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
    with tf.io.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token, _ = token.split("\t")
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

def triangle_eval(span1,span2,span3,span4):
    for indx in range(0, len(span4)):
        try:
            if span4[indx] == 1:
                span3[indx] = span3[indx+1] = span3[indx+2] = span3[indx+3] = 0
                span2[indx] = span2[indx+1] = span2[indx+2] = span2[indx+3] = 0
                span1[indx] = span1[indx+1] = span1[indx+2] = span1[indx+3] = 0
                continue
            else:
                if span3[indx] == 1:
                    span2[indx] = span2[indx+1] = span2[indx+2] = 0
                    span1[indx] = span1[indx+1] = span1[indx+2] = 0
                    continue
                else:
                    if span2[indx] == 1:
                        span1[indx] = span1[indx+1] = 0
                        continue
        except IndexError:
            pass
    return span1, span2, span3, span4

def arrange_ann(annotations, Entdic, EntityList, Triggerdic, Reldic, relation_mapping, rel_index, rel_dic):
    for line in annotations:
        instance = line.split('\t')
        if 'T' in instance[0]: #instance=['T0', 'Machining 4 9', '上向き削り']
            instance_tag = instance[1].split(' ')[0] #instance_tag = 'Machining'
            if 'Trigger' not in instance_tag: #TriggerタグではなかったらEntitiy
                Entdic[instance[0]] = (instance[1].split()[0],int(instance[1].split()[1]),int(instance[1].split()[2]),instance[2])
                EntityList.append(instance[0])
                #Entdic[T0] = (Machihing, 4, 9, '上向き削り') 
            else:
                Triggerdic[instance[0]] = (instance[1].split()[0],int(instance[1].split()[1]),int(instance[1].split()[2]),instance[2])

        elif 'R' in instance[0]: #instance = ['R1', 'Relation Arg1:T30 Arg2:T25', '']
            instance_tag = instance[1].split(' ')[0]
            Reldic[instance[0]] = (instance[1].split(' ')[0],instance[1].split(' ')[1][5:],instance[1].split(' ')[2][5:])
            if instance_tag == "UpDown" or instance_tag == "DownUp":
                continue

            if instance_tag in relation_mapping:
                instance_tag = relation_mapping[instance_tag]

            if instance_tag not in rel_dic:
                rel_dic[instance_tag] = rel_index
                rel_index += 1
    return Entdic, EntityList, Triggerdic, Reldic, relation_mapping, rel_index, rel_dic

def nest_process(Entdic, data_structure_type):
    if data_structure_type == "NEST_OUTER":
        removed_Entdic = dict()
        removed_EntityList = []
        num_of_entity = 0
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
                
        TARGET_DIC = removed_Entdic
        num_of_entity = len(removed_Entdic)
        return TARGET_DIC, num_of_entity

    if data_structure_type == "NEST_INNER":
        TARGET_DIC = Entdic
        num_of_entity = len(Entdic)
        return TARGET_DIC, num_of_entity

    else:
        raise ValueError('CHOOSE DATA STRUCTURE ! (NEST_INNER or NEST_OUTER)')





def main(config):
    max_length = 150
    Rel_None_Label = config.get('makedata', 'REL_NONE_LABEL')
    relation_mapping = {'NegativeLinear':'Negative', 'Linear':'Relation', 'Square':'Relation'}
    
    vocab = load_vocab('../spm_model/wiki-ja.vocab')
    sp = spm.SentencePieceProcessor()
    sp.Load('../spm_model/wiki-ja.model')

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--structure',default = "NEST_INNER", help='NEST_INNER -> All nest entity\nNEST_OUTER -> only out nest entity')
    parser.add_argument('--target_type',default = "Relation", help='All        -> all sentence   \nRelation   -> only relation sentence\nRelation_trigger  -> only relatin sentence and thinking trigger')
    parser.add_argument('--span_size', default = 4, help = 'Choose span size \ndefault = 4')
    
    data_structure_type = parser.parse_args().structure
    target_data_type = parser.parse_args().target_type
    span_size = int(parser.parse_args().span_size)


    data_strucuture_type_list = ["NEST_INNER", "NEST_OUTER"]
    target_data_type_list = ["All", "Relation", "Relation_trigger"]

    assert data_structure_type in data_strucuture_type_list
    assert target_data_type in target_data_type_list


    data_type_list = ["train_devel","test"]

    for data in data_type_list:
        if data == "train_devel":
            data_path = config.get('preprocess path', 'DATA_PATH_TRAIN_DEVEL')
            if target_data_type == "All":
                dataname = str(span_size) + config.get('dataname', 'N_REL_DIVIDED_TRAIN_DEVEL')
            if target_data_type == "Relation":
                dataname = str(span_size) + config.get('dataname', 'N_REL_DIVIDED_TRAIN_DEVEL_SHORT')
            if target_data_type == 'Relation_trigger':
                dataname = str(span_size) + config.get('dataname', 'N_REL_DIVIDED_TRAIN_DEVEL_TRIGGER')
        if data == "test":
            data_path = config.get('preprocess path', 'DATA_PATH_TEST')
            if target_data_type == "All":
                dataname = str(span_size) + config.get('dataname', 'N_REL_DIVIDED_TEST')
            if target_data_type == "Relation":
                dataname = str(span_size) + config.get('dataname', 'N_REL_DIVIDED_TEST_SHORT')
            if target_data_type == 'Relation_trigger':
                dataname = str(span_size) + config.get('dataname', 'N_REL_DIVIDED_TEST_TRIGGER')



        filelst = os.listdir(data_path)
        doclist = []
        rel_dic = dict()  #全体の関係辞書
        rel_dic[Rel_None_Label] = 0
        REL_LABEL_DICT = {}

        rel_index = 1
        final_num_of_entity = 0
        Num_of_data = 0
        Num_of_include_trigger = 0
        Num_of_miss_tokenize = 0
        miss_tokenize_list = [0]
        hitotsu = 0
        kuttuki = 0
        nest_entity_number_list = [0 for _ in range(span_size)]
        corpus = []

        filename_lst = []


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
            filename_lst.append(doc[0][:-4])
            Entdic = dict()
            EntityList = []
            Reldic = dict()
            REL_LABEL_unit = []
            Triggerdic = dict()

            with open(data_path + doc[0], 'r', encoding='utf-8') as txtfile:
                text = txtfile.read()

            with open(data_path + doc[1], 'r', encoding='utf-8') as annfile:
                annotations = annfile.read().split('\n')
            
            #------------------------------------------ ann ファイル内を処理．辞書作成 -----------------------------------------------
            Entdic, EntityList, Triggerdic, Reldic, relation_mapping, rel_index, rel_dic = arrange_ann(annotations, Entdic, EntityList, Triggerdic, Reldic, relation_mapping, rel_index, rel_dic)
            if  "short" in dataname:
                if len(Reldic) == 0:
                    continue
                else:
                    if len(Triggerdic) > 0:
                        Num_of_include_trigger += 1


            #------------------------------------------- NEST 構造を処理するかしないか --------------------------------------------------------------------------
            TARGET_DIC, num_of_entity = nest_process(Entdic, data_structure_type)
            final_num_of_entity += num_of_entity


            #--------------------------------------------- Tokenize ------------------------------------------------------------------------
            tokenized = sp.EncodeAsPieces(text)



            if len(tokenized[0]) == 1:
                tokenized[0] = '[CLS]'
                hitotsu += 1
            else:
                tokenized[0] = tokenized[0][1:]
                tokenized.insert(0, '[CLS]')
                kuttuki += 1

            tokenized.append('[SEP]')
            attention_mask = [1] * len(tokenized)
            tag_dict = {} # tag_dict['T1631'] = (outfilmのサイズ, 何トークン目か)
            cls_tag_dict = {} #上のtag_dictは何トークン目かで最初にCLSトークンが足されることが想定されていないためこの辞書で補正

            target_spm = []
            start_cnt = 0
            end_cnt = 0

            for x, subword in enumerate(tokenized[1:-1]):
                end_cnt = len(subword)
                target_spm.append((x, subword, start_cnt, start_cnt + end_cnt))
                start_cnt += end_cnt


            # 各spanにおけるラベルの作成
            output_fiims = [[0] * (max_length - 1) for s_n in range(0,span_size)]
            for s_y in range(0,span_size):
                for t in range(len(target_spm)):
                    output_fiims[s_y][t] = 0
                    for k, v in TARGET_DIC.items():
                        try:
                            if target_spm[t][2] == v[1] and target_spm[t + s_y][3] == v[2]:
                                output_fiims[s_y][t] = 1
                                nest_entity_number_list[s_y] += 1
                                tag_dict[k] = (s_y+1,target_spm[t])
                        except IndexError:
                            pass

            # trigger vecの作成

            trigger_vec = [0] * (max_length - 1)
            for t in range(len(target_spm)):
                for k,v in Triggerdic.items():
                    try:
                        if target_spm[t][2] <= v[1] < target_spm[t][3] or target_spm[t][2] < v[2] <= target_spm[t][3]:
                            trigger_vec[t] = 1
                    except IndexError:
                        pass
            trigger_vec.insert(0,0)


            # [CLS]の処理
            for one in output_fiims:
                one.insert(0,0)
            for k_tag, v_tag in tag_dict.items():
                cls_tag_dict[k_tag] = (v_tag[0],(v_tag[1][0]+1,v_tag[1][1],v_tag[1][2],v_tag[1][3]))


            # [PAD]の処理
            for i in range(max_length - len(tokenized)):
                tokenized.append('[PAD]')
                attention_mask.append(0)

            indx_tokens = [vocab[s] if s in vocab else vocab['<unk>'] for s in tokenized]


            for rel_v in Reldic.values():
                try:
                    REL_LABEL_unit.append((rel_dic[rel_v[0]],(cls_tag_dict[rel_v[1]],cls_tag_dict[rel_v[2]])))
                except:
                    pass
            REL_LABEL_DICT[n] = REL_LABEL_unit

            corpus.append(((n,doc,Entdic,Reldic), indx_tokens, output_fiims, attention_mask, tokenized, trigger_vec))
            Num_of_data += 1
        pdb.set_trace()
        print('Number of sentence ' + str(Num_of_data))
        print('_ num is {0}'.format(hitotsu))
        print('_*** num is {0}'.format(kuttuki))
        for s_y in range(span_size):
            print('Number of size {0} entity is {1}'.format(s_y+1,nest_entity_number_list[s_y]))
        
        print('Number of entity including NEST structure {0}'.format(final_num_of_entity))
        print('Number of Miss tokenize is {}'.format(Num_of_miss_tokenize))
        print('Number of data of including trigger word {0}'.format(Num_of_include_trigger))
        database = shelve.open(config.get('preprocess path', 'SHELVE_PATH'))
        database[dataname] = [vocab, rel_dic, corpus, filename_lst,REL_LABEL_DICT]
        database.close()

        if data == 'train_devel':
            REL_database = shelve.open(config.get('preprocess path', 'REL_DIC_PATH'))
            REL_database['rel_dic'] = rel_dic
            REL_database.close()





if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('../../machine_BRAN.conf')
    main(config)
