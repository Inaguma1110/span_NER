import torch
import torch.nn as nn
import numpy as np

import argparse
import os
import pdb
import shelve
import shutil
import configparser


def make_dir_and_write_annfile(pred_log_dir, epoch, name, txt, ann, rel_ann):
    os.makedirs('{0}/{1}'.format(pred_log_dir,epoch), exist_ok=True)
    txtpred = open('{0}/{1}/pred_{2}'.format(pred_log_dir,epoch, name[0]), 'w')
    txtpred.write(''.join(txt))
    txtpred.close()

    annpred = open('{0}/{1}/pred_{2}'.format(pred_log_dir,epoch, name[1]), 'w')
    annpred.write(''.join(ann))
    annpred.write(''.join(rel_ann))
    annpred.close()
    
def pred_rel2ann(epoch, pred_log_dir, doc_dict, n_docs, pred_tokens, pred_span1, pred_span2, pred_span3, pred_span4, rel_ann_dict, REL_DIC):
    print('writing predicts text and annotation ... ')
    continue_list = ['[CLS]','[SEP]','[PAD]'] 
    for n_sentence in range(0, len(pred_tokens)):
        name,entdic,reldic = doc_dict[n_docs[n_sentence]]
        relindex_pairs = rel_ann_dict[n_docs[n_sentence]]
        # pdb.set_trace()
        txt = []
        ann = []
        rel_ann = []
        tag_dict ={}
        w_start = 0
        tag_cnt = 0
        rel_tag_cnt = 0
        for i_token in range(0, len(pred_tokens[n_sentence])):
            if pred_tokens[n_sentence][i_token] in continue_list:
                continue
            w_len = len(pred_tokens[n_sentence][i_token])

            token_pred_span1 = pred_span1[n_sentence][i_token]
            token_pred_span2 = pred_span2[n_sentence][i_token]
            token_pred_span3 = pred_span3[n_sentence][i_token]
            token_pred_span4 = pred_span4[n_sentence][i_token]
            
            if token_pred_span1 == 1:
                token = pred_tokens[n_sentence][i_token]
                token_ann = ("T{0}\t{1} {2} {3}\t{4}\n".format(tag_cnt, 'Machining', w_start, w_start + len(token), token))
                ann.append(token_ann)
                tag_dict[(1,i_token)] = "T{}".format(tag_cnt)
                tag_cnt += 1

            if token_pred_span2 == 1:
                token = pred_tokens[n_sentence][i_token] + pred_tokens[n_sentence][i_token+1]
                token_ann = ("T{0}\t{1} {2} {3}\t{4}\n".format(tag_cnt, 'Machining', w_start, w_start + len(token), token))
                ann.append(token_ann)
                tag_dict[(2,i_token)] = "T{}".format(tag_cnt)
                tag_cnt += 1

            if token_pred_span3 == 1:
                token = pred_tokens[n_sentence][i_token] + pred_tokens[n_sentence][i_token+1] + pred_tokens[n_sentence][i_token+2]
                token_ann = ("T{0}\t{1} {2} {3}\t{4}\n".format(tag_cnt, 'Machining', w_start, w_start + len(token), token))
                ann.append(token_ann)
                tag_dict[(3,i_token)] = "T{}".format(tag_cnt)
                tag_cnt += 1

            if token_pred_span4 == 1:
                token = pred_tokens[n_sentence][i_token] + pred_tokens[n_sentence][i_token+1] + pred_tokens[n_sentence][i_token+2] + pred_tokens[n_sentence][i_token+3]
                token_ann = ("T{0}\t{1} {2} {3}\t{4}\n".format(tag_cnt, 'Machining', w_start, w_start + len(token), token))
                ann.append(token_ann)
                tag_dict[(4,i_token)] = "T{}".format(tag_cnt)
                tag_cnt += 1
            
            txt.append(pred_tokens[n_sentence][i_token])
            w_start += w_len
        for rel_unit in relindex_pairs:
            rel_index = rel_unit[0]
            pair_unit_head = tuple(rel_unit[1][0])
            pair_unit_tail = tuple(rel_unit[1][1])
            if pair_unit_head in tag_dict and pair_unit_tail in tag_dict:
                rel_ann.append("R{0}\t{1} Arg1:{2} Arg2:{3}\n".format(rel_tag_cnt, [k for k, v in REL_DIC.items() if v == rel_index][0],tag_dict[pair_unit_head],tag_dict[pair_unit_tail]))
                rel_tag_cnt+=1
        make_dir_and_write_annfile(pred_log_dir, epoch, name, txt, ann, rel_ann)


    print('finish \n')
        # pdb.set_trace()

# entity を記述したannファイルに追加でRelationの情報を記述する
# def relation_function(epoch, pred_log_dir, doc_dict, n_docs, pred_tokens, pred_span1, pred_span2, pred_span3, pred_span4):
#     pass

#     annpred = open('{0}/{1}/pred_{2}'.format(pred_log_dir,epoch, name[1]), 'a')
#     annpred.write(''.join(ann))
#     annpred.close()