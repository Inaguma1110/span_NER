import torch
import torch.nn as nn
import numpy as np

import argparse
import os
import pdb
import shelve
import shutil
import configparser


def make_dir_and_write_annfile(pred_log_dir, epoch, name, txt, ann):
    os.makedirs('{0}/{1}'.format(pred_log_dir,epoch), exist_ok=True)
    txtpred = open('{0}/{1}/pred_{2}'.format(pred_log_dir,epoch, name[0]), 'w')
    txtpred.write(''.join(txt))
    txtpred.close()

    annpred = open('{0}/{1}/pred_{2}'.format(pred_log_dir,epoch, name[1]), 'w')
    annpred.write(''.join(ann))
    annpred.close()
    
def entity_function(epoch, pred_log_dir, doc_dict, n_docs, pred_tokens, pred_span1, pred_span2, pred_span3, pred_span4):
    # pdb.set_trace()
    print('writing predicts text and annotation ... ')
    continue_list = ['[CLS]','[SEP]','[PAD]'] 
    for n_sentence in range(0, len(pred_tokens)):
        name,entdic,reldic = doc_dict[n_docs[n_sentence]]
        # pdb.set_trace()
        txt = []
        ann = []
        w_start = 0
        tag_cnt = 0

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
                tag_cnt += 1

            if token_pred_span2 == 1:
                token = pred_tokens[n_sentence][i_token] + pred_tokens[n_sentence][i_token+1]
                token_ann = ("T{0}\t{1} {2} {3}\t{4}\n".format(tag_cnt, 'Machining', w_start, w_start + len(token), token))
                ann.append(token_ann)
                tag_cnt += 1

            if token_pred_span3 == 1:
                token = pred_tokens[n_sentence][i_token] + pred_tokens[n_sentence][i_token+1] + pred_tokens[n_sentence][i_token+2]
                token_ann = ("T{0}\t{1} {2} {3}\t{4}\n".format(tag_cnt, 'Machining', w_start, w_start + len(token), token))
                ann.append(token_ann)
                tag_cnt += 1

            if token_pred_span4 == 1:
                token = pred_tokens[n_sentence][i_token] + pred_tokens[n_sentence][i_token+1] + pred_tokens[n_sentence][i_token+2] + pred_tokens[n_sentence][i_token+3]
                token_ann = ("T{0}\t{1} {2} {3}\t{4}\n".format(tag_cnt, 'Machining', w_start, w_start + len(token), token))
                ann.append(token_ann)
                tag_cnt += 1
            
            txt.append(pred_tokens[n_sentence][i_token])
            w_start += w_len
        
        # pdb.set_trace()
        make_dir_and_write_annfile(pred_log_dir, epoch, name, txt, ann)
        # os.makedirs('{0}/{1}'.format(pred_log_dir,epoch), exist_ok=True)
        # txtpred = open('{0}/{1}/pred_{2}'.format(pred_log_dir,epoch, name[0]), 'w')
        # txtpred.write(''.join(txt))
        # txtpred.close()

        # annpred = open('{0}/{1}/pred_{2}'.format(pred_log_dir,epoch, name[1]), 'w')
        # annpred.write(''.join(ann))
        # annpred.close()
    print('finish \n')
    # shutil.copy('../brat_visual/visual.conf', '{0}/{1}'.format(pred_log_dir, epoch))
    # shutil.copy('../brat_visual/annotation.conf', '{0}/{1}'.format(pred_log_dir, epoch))
    
        # pdb.set_trace()

# entity を記述したannファイルに追加でRelationの情報を記述する
def relation_function(epoch, pred_log_dir, doc_dict, n_docs, pred_tokens, pred_span1, pred_span2, pred_span3, pred_span4):
    pass

    annpred = open('{0}/{1}/pred_{2}'.format(pred_log_dir,epoch, name[1]), 'a')
    annpred.write(''.join(ann))
    annpred.close()