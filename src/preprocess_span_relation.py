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
    data_output_type = config.get('makedata', 'DATATYPE')
    filelst = os.listdir(data_path)
    doclist = []

    sp = spm.SentencePieceProcessor()
    sp.Load('./spm_model/wiki-ja.model')
    tokenizer = BertTokenizer.from_pretrained('./spm_model/wiki-ja.vocab.txt')

    REL_DIC = dict()  #全体の関係辞書
    REL_DIC['None'] = 0
    rel_index = 1

    Num_of_data = 0
    Num_of_entity = 0

    hitotsu = 0
    kuttuki = 0
    Numspan1 = 0
    Numspan2 = 0
    Numspan3 = 0
    Numspan4 = 0
    Numspan5 = 0
    Numspan6 = 0
    Numspan7 = 0
    Numspan8 = 0
    Numspan9 = 0
    Numspan10 = 0
    Numspan11 = 0
    
    corpus = []

    filename_lst = []

    # dataname = 'Machining_data_kakogijutsu_span_NER_RE_pretrain_Japanese_BERT_maxlength_100'
    dataname = config.get('dataname', 'SPAN_TOKENIZED')
    print('\n save dataname is {0}\n'.format(dataname))
    print('\n target data type is {0}\n'.format(data_output_type))
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
        # doc = ('seimitu_0022_389.txt', 'seimitu_0022_389.ann')
        filename_lst.append(doc[0][:-4])
        Entdic = dict()
        removed_Entdic = dict()
        Reldic = dict()
        Triggerdic = dict()
        vocab = load_vocab('./spm_model/wiki-ja.vocab')

        TARGET_DIC = {}
        
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

        Num_of_entity += len(Entdic)


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
        # Num_of_entity += len(removed_Entdic)
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

        target_spm          = [] 
        start_cnt = 0
        end_cnt   = 0
        for x, subword in enumerate(spmed[1:-1]):
            end_cnt = len(subword)
            target_spm.append((x, subword, start_cnt, start_cnt + end_cnt))
            start_cnt += end_cnt


        tokens = []
        target_tokens = []
        unit = ''
        unit_total = 0
        token_cnt = 0

        # pdb.set_trace()
        for character in text:
            unit_flag = False
            unit_total += 1
            # pdb.set_trace()
            for ent_match in Entdic.values():
                if unit_total == int(ent_match[1]) or unit_total == int(ent_match[2]):
                    unit_flag = True
            # pdb.set_trace()
            for trig_match in Triggerdic.values():
                if unit_total == int(trig_match[1]) or unit_total == int(trig_match[2]):
                    unit_flag = True
            # pdb.set_trace()
            for spm_match in target_spm:
                if unit_total == spm_match[2] or unit_total == spm_match[3]:
                    unit_flag = True

            # pdb.set_trace()
            unit += character
            if unit_flag:
                tokens.append(unit)
                target_tokens.append((unit, token_cnt, token_cnt + len(unit)))
                token_cnt += len(unit)
                unit = ''

        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')

        # pdb.set_trace()

        #Paddingは0で置き換え
        output_film_size1   = [0] * (max_length - 1)
        output_film_size2   = [0] * (max_length - 1)
        output_film_size3   = [0] * (max_length - 1)
        output_film_size4   = [0] * (max_length - 1)

        attention_mask      = [1] * len(spmed)

        for x1 in range(len(target_tokens)):
            x2 = x1 + 1
            x3 = x1 + 2
            x4 = x1 + 3
            x5 = x1 + 4
            x6 = x1 + 5
            x7 = x1 + 6 
            x8 = x1 + 7 
            x9 = x1 + 8 
            x10 = x1 + 9
            x11 = x1 + 10 


            # subwordが存在するindexは0に置き換える
            output_film_size1[x1] = 0
            output_film_size2[x1] = 0
            output_film_size3[x1] = 0
            output_film_size4[x1] = 0


            try:
                if data_output_type == 'NEST_INNER':
                    TARGET_DIC = Entdic
                elif data_output_type == 'LARGE':
                    TARGET_DIC = removed_Entdic
                else:
                    raise ValueError('Choose datatype')
            except ValueError as e:
                print(e)

            for k, v in TARGET_DIC.items():      #Entityの存在する部分にfilter毎に1を立てる
                try:
                    if target_tokens[x1][1] == v[1] and target_tokens[x1][2] == v[2]:
                        output_film_size1[x1] = 1
                        Numspan1 += 1

                    if target_tokens[x1][1] == v[1] and target_tokens[x2][2] == v[2]:
                        output_film_size2[x1] = 1
                        Numspan2 += 1

                    if target_tokens[x1][1] == v[1] and target_tokens[x3][2] == v[2]:
                        output_film_size3[x1] = 1
                        Numspan3 += 1

                    if target_tokens[x1][1] == v[1] and target_tokens[x4][2] == v[2]:
                        output_film_size4[x1] = 1
                        Numspan4 += 1


                    if target_tokens[x1][1] == v[1] and target_tokens[x5][2] == v[2]:
                        Numspan5 += 1

                    if target_tokens[x1][1] == v[1] and target_tokens[x6][2] == v[2]:
                        Numspan6 += 1

                    if target_tokens[x1][1] == v[1] and target_tokens[x7][2] == v[2]:
                        Numspan7 += 1

                    if target_tokens[x1][1] == v[1] and target_tokens[x8][2] == v[2]:
                        Numspan8 += 1

                    if target_tokens[x1][1] == v[1] and target_tokens[x9][2] == v[2]:
                        Numspan9 += 1

                    if target_tokens[x1][1] == v[1] and target_tokens[x10][2] == v[2]:
                        Numspan10 += 1

                    if target_tokens[x1][1] == v[1] and target_tokens[x11][2] == v[2]:
                        Numspan11 += 1

                except IndexError:
                    pass

        #[CLS]の分insert
        output_film_size1.insert(0,0)
        output_film_size2.insert(0,0)
        output_film_size3.insert(0,0)
        output_film_size4.insert(0,0)
        # attention_mask[0] = 1

        for i in range(max_length - len(tokens)):
            tokens.append('[PAD]')
            attention_mask.append(0)
        # pdb.set_trace()

        indx_tokens = [vocab[s] if s in vocab else vocab['<unk>'] for s in tokens]


        # pdb.set_trace()

        corpus.append((doc[0], indx_tokens, output_film_size1, output_film_size2, output_film_size3, output_film_size4, attention_mask, tokens, (n,doc,removed_Entdic, Reldic)))

        Num_of_data += 1
        Numallspan = Numspan1 + Numspan2 + Numspan3 + Numspan4 + Numspan5 + Numspan6 + Numspan7 + Numspan8 + Numspan9 + Numspan10 + Numspan11
        #pdb.set_trace()

    print('_ only separated num is {0}'.format(hitotsu))
    print('_*** not separated num is {0}'.format(kuttuki))
    print('Number of data ' + str(Num_of_data))
    print('Number of entity including NEST structure {0}'.format(Num_of_entity))
    print('Number of span1 is {0}  {1} percent  {2} cummurative percent'.format(Numspan1,(Numspan1/Numallspan),(Numspan1/Numallspan)))
    print('Number of span2 is {0}  {1} percent  {2} cummurative percent'.format(Numspan2,(Numspan2/Numallspan),(Numspan1+Numspan2)/Numallspan))
    print('Number of span3 is {0}  {1} percent  {2} cummurative percent'.format(Numspan3,(Numspan3/Numallspan),(Numspan1+Numspan2+Numspan3)/Numallspan))
    print('Number of span4 is {0}  {1} percent  {2} cummurative percent'.format(Numspan4,(Numspan4/Numallspan),(Numspan1+Numspan2+Numspan3+Numspan4)/Numallspan))
    print('Number of span5 is {0}  {1} percent  {2} cummurative percent'.format(Numspan5,(Numspan5/Numallspan),(Numspan1+Numspan2+Numspan3+Numspan4+Numspan5)/Numallspan))
    print('Number of span6 is {0}  {1} percent  {2} cummurative percent'.format(Numspan6,(Numspan6/Numallspan),(Numspan1+Numspan2+Numspan3+Numspan4+Numspan5+Numspan6)/Numallspan))
    print('Number of span7 is {0}  {1} percent  {2} cummurative percent'.format(Numspan7,(Numspan7/Numallspan),(Numspan1+Numspan2+Numspan3+Numspan4+Numspan5+Numspan6+Numspan7)/Numallspan))
    print('Number of span8 is {0}  {1} percent  {2} cummurative percent'.format(Numspan8,(Numspan8/Numallspan),(Numspan1+Numspan2+Numspan3+Numspan4+Numspan5+Numspan6+Numspan7+Numspan8)/Numallspan))
    print('Number of span9 is {0}  {1} percent  {2} cummurative percent'.format(Numspan9,(Numspan9/Numallspan),(Numspan1+Numspan2+Numspan3+Numspan4+Numspan5+Numspan6+Numspan7+Numspan8+Numspan9)/Numallspan))
    print('Number of span10 is {0}  {1} percent  {2} cummurative percent'.format(Numspan10,(Numspan10/Numallspan),(Numspan1+Numspan2+Numspan3+Numspan4+Numspan5+Numspan6+Numspan7+Numspan8+Numspan9+Numspan10)/Numallspan))
    print('Number of span11 is {0}  {1} percent  {2} cummurative percent'.format(Numspan11,(Numspan11/Numallspan),(Numspan1+Numspan2+Numspan3+Numspan4+Numspan5+Numspan6+Numspan7+Numspan8+Numspan9+Numspan10+Numspan11)/Numallspan))

    database = shelve.open(config.get('path', 'SHELVE_PATH'))
    database[dataname] = [vocab, REL_DIC, corpus, filename_lst]

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('../machine.conf')
    main(config)
