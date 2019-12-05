import sys
import MeCab as mcb
import re
import six
import os
import configparser
import pdb
import shelve
import numpy as np
import itertools
import tqdm
import sys, codecs
import collections
import sentencepiece as spm

import tensorflow as tf
from pytorch_transformers import BertTokenizer, BertModel, BertConfig

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


def sentencepiece_tokenize(text):
    sp = spm.SentencePieceProcessor()
    sp.Load('./spm_model/wiki-ja.model')
    return sp.EncodeAsPieces(text)

def tokenize(doc):
    mcbtagger = mcb.Tagger("-Ochasen")
    tagger = mcbtagger
    data = tagger.parse(doc)
    return data

def NER_BIOtagger(ordercnt, target, dic):
    target_end = ordercnt + len(target)
    for v in dic.values():
        dicstart = v[1]
        dicend = v[2]
        if ordercnt == dicstart: #Bのタグが付けられるかチェック
            if target_end < dicend: 
                return 'B'
            elif target_end == dicend:
                return 'B'
            #elif target_end > dicend:
            #    return 'Error'
        elif ordercnt > dicstart: #Iのタグが付けられるかチェック
            if target_end < dicend:
                return 'I'
            elif target_end == dicend:
                return 'I'
            #elif target_end > dicend:
            #    return 'Error'
    return 'O'

def BIOtagger(order_sta, order_end, entity_dic, pair):     #ペアとするEntity以外はOtherを出力するBIOのタグ付け
    no_target_type = 'Other'
    target1 = entity_dic[pair[0]][0]
    target2 = entity_dic[pair[1]][0]

    for key,value in entity_dic.items():
        tag = value[0]
        start = value[1]
        end = value[2]
        surface = value[3]

        if start == order_sta and order_end <= end:
            if key in pair and (target1 == tag or target2 == tag):
                return 'B-' + entity_dic[key][0]
            else:
                return no_target_type

        elif start < order_sta  and  order_end <= end:
            if key in pair and (target1 == tag or target2 == tag):
                return 'I-' + entity_dic[key][0]
            else:
                return no_target_type
    return 'O'


def BIO_Machining_tagger(order_sta, order_end, entity_dic, pair):     #ペアとするEntity以外はOtherを出力するBIOのタグ付け
    no_target_type = 'Other'
    target1 = entity_dic[pair[0]][0]
    target2 = entity_dic[pair[1]][0]

    for key,value in entity_dic.items():
        tag = value[0]
        start = value[1]
        end = value[2]
        surface = value[3]

        if start == order_sta and order_end <= end:
            if key in pair and (target1 == tag or target2 == tag):
                return 'B-Machining'
            else:
                return no_target_type

        elif start < order_sta  and  order_end <= end:
            if key in pair and (target1 == tag or target2 == tag):
                return 'I-Machining'
            else:
                return no_target_type
    return 'O'


def main(config):
    data_path = config.get('path', 'DATA_PATH')
    max_sent_len      = int(config.get('makedata', 'MAX_SENT_LEN'))
    filelst = os.listdir(data_path)
    doclist = []
    REL_DIC = dict()  #全体の関係辞書
    REL_DIC['Non'] = 0
    TAG_DIC = dict()  #全体のBIOタグ辞書
    rel_index = 1
    tag_index = 1
    vocab = load_vocab('./spm_model/wiki-ja.vocab')

    Num_B_tag = 0
    Num_I_tag = 0
    Num_O_tag = 0
    Num_of_data = 0
    Num_of_entity = 0

    corpus = []
    filename_lst = []

    new_filelst = []

    dataname = config.get('dataname', 'BIO')

    if dataname == 'Machining_data_kakogijutsu': # seimitsuとendmillファイルは対象外とする．
        for fname in filelst:
            if "seimitu" in fname or "endmil" in fname:
                continue
            new_filelst.append(fname)    
    else:
        new_filelst = filelst


    for txtfile in new_filelst:
        docpair = ()
        if (txtfile[-4:]) == '.txt':
            for annfile in new_filelst:
                if (annfile[-4:]) == '.ann' and annfile[:-4] == txtfile[:-4]:
                    docpair = (txtfile, annfile)
                    doclist.append(docpair)


    # pdb.set_trace()
    noisedoclst = []

    for i, doc in enumerate(tqdm.tqdm(doclist)):
        if doc in noisedoclst:
            continue
        # print('create data of ' + str(doc[0]))
        filename_lst.append(doc[0][:-4])
        #pdb.set_trace()
        Entdic = dict()
        removed_Entdic = dict()
        Reldic = dict()
        Triggerdic = dict()

        with open(data_path + doc[0], 'r', encoding='utf-8') as txtfile:
            text = txtfile.read()

        with open(data_path + doc[1], 'r', encoding='utf-8') as annfile:
            annotations = annfile.read().split('\n')
        
        #pdb.set_trace()
        # annファイル処理．anotationを全て辞書に保存する処理
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
        #pdb.set_trace()

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
        Num_of_entity += len(removed_Entdic)
        #################################################################################################################
        
        # pdb.set_trace()
        
        # mecabed = tokenize(text).split('\n')
        spmed = sentencepiece_tokenize(text)
        # pdb.set_trace()

        sentence_tag_map    = [0] * (max_sent_len-1)


        target_spm = []
        start_cnt = 0
        end_cnt   = 0

        for x, token in enumerate(spmed[1:]):
            end_cnt = len(token)
            target_spm.append((x, token, start_cnt, start_cnt + end_cnt))
            start_cnt += end_cnt

        # pdb.set_trace()

        for j, unit in enumerate(target_spm):
            # pdb.set_trace()
            if unit[1] == 'EOS':
                continue
            if unit[1] == '':
                continue

            BIOtag = NER_BIOtagger(unit[2],unit[1],removed_Entdic)

            if BIOtag not in TAG_DIC:
                TAG_DIC[BIOtag] = tag_index
                tag_index += 1
            if BIOtag[0] == 'B':
                Num_B_tag += 1
            if BIOtag[0] == 'I':
                Num_I_tag += 1
            if BIOtag[0] == 'O':
                Num_O_tag += 1
            #pdb.set_trace()

            sentence_tag_map[j] = TAG_DIC[BIOtag]
        #     if BIOtag[0] == 'B' or BIOtag[0] == 'I':
        #         ent_position_map[x] = 1
        spmed[0] = '[CLS]'
        sentence_tag_map.insert(0,TAG_DIC['O'])
        spmed.append('[SEP]')
        attention_mask      = [1] * len(spmed)

        while len(spmed) < max_sent_len:
            spmed.append('[PAD]')
            attention_mask.append(0)


        indx_tokens = [vocab[s] if s in vocab else vocab['<unk>'] for s in spmed]


        #if doc[0] == "**********.txt":
        # pdb.set_trace()

        corpus.append((doc[0], indx_tokens, sentence_tag_map, attention_mask, spmed, removed_Entdic, Reldic))

    Num_of_data += (i+1)
    # print(str(i+1) + ' data created from ' + str(doc[1][:-4]))
    # print()
    #pdb.set_trace()

    print(noisedoclst)
    print(str(len(noisedoclst)) + ' removed')
    Num_all_tag = Num_B_tag + Num_I_tag + Num_O_tag
    print(str('\nFinished BIO tagging\n'))
    print('Number of B tag ' , Num_B_tag ,'{:.2f}'.format((Num_B_tag/Num_all_tag) * 100) ,'%' )
    print('Number of I tag ' , Num_I_tag ,'{:.2f}'.format((Num_I_tag/Num_all_tag) * 100) ,'%' )
    print('Number of O tag ' , Num_O_tag ,'{:.2f}'.format((Num_O_tag/Num_all_tag) * 100) ,'%' )
    print('Number of removed nest entity {0}'.format(Num_of_entity))
    print('Number of data ' + str(Num_of_data))
    #pdb.set_trace()

    database = shelve.open(config.get('path', 'SHELVE_PATH'))
    database[dataname] = [vocab, corpus, TAG_DIC, filename_lst]

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('../machine.conf')
    main(config)
