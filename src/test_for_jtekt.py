import sys
import collections
import six
import argparse
import pdb
import tqdm, contextlib
import time
import shelve
import configparser
import datetime
import sentencepiece as spm
import os
from progressbar import progressbar
from distutils.util import strtobool
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as D
from torch.utils.tensorboard import SummaryWriter

import tensorboardX as tb
import tensorflow as tf

from model.BRAN_old import BERT_PRETRAINED_MODEL_JAPANESE, SPAN_CNN, PAIRS_MODULE, RELATION, BRAN
from util import pred_rel2ann,pred2text,nest_entity_process


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



print('\nCreate Environment...\n')

## reading config file #############################################################
config = configparser.ConfigParser()
config.read('../machine_BRAN.conf')
dt_now = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

max_sent_len      = int(config.get('makedata', 'MAX_SENT_LEN'))
weight            = config.get('CNNs', 'WEIGHT')
batch             = config.get('main', 'BATCH_SIZE_TRAIN')
lr_str            = config.get('main', 'LEARNING_RATE')
attention_mask_is = config.get('CNNs', 'ATTENTION_MASK_IS')
TARGET_only_LARGE_NEST_flag = strtobool(config.get('main','NEST'))
NER_model_save_path = config.get('model path', 'oldNERmodel')
RE_model_save_path = config.get('model path', 'oldREmodel')

####################################################################################

## writer setting ##################################################################
writer_log_dir ='../../data/TensorboardGraph/span_NER_RE/Test/'+dt_now+'_for_JTEKT'
brat_log_dir   = '../../brat/brat-v1.3_Crunchy_Frog/data/model_preds/span_NER_RE/Test/'+dt_now+'_for_JTEKT'
model_writer_log_dir ='../../data/model/'
####################################################################################

np.random.seed(1)
torch.manual_seed(1)
# pdb.set_trace()

data_path = config.get('path', 'FOR_JTEKT')
target_path = data_path + 'targetdata/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('finish', end='\n')
print('\nCreate data...')
filelst = os.listdir(data_path)
txtlst = []
for i_f in filelst:
    if i_f[-4:] == '.txt':
        txtlst.append(i_f)
# -----------------------------make entity dictionary and relation dictionary------------------------------------------
for x in range(len(txtlst)):
    begtotal = 0
    endtotal = 0
    num = 0
    txtfile = open(data_path + txtlst[x], "r")
    for k, tlineout in enumerate(txtfile):
        f = open(target_path + txtlst[x][:-4] + "_" + "{:0=3}".format(k) + ".txt", "w")
        f.write(tlineout)
        f.close()

doclist = []
vocab = load_vocab('./spm_model/wiki-ja.vocab')
sp = spm.SentencePieceProcessor()
sp.Load('./spm_model/wiki-ja.model')

RELdatabase = shelve.open(config.get('path', 'REL_DIC_PATH'))
REL_DIC = RELdatabase['REL_DIC']
RELdatabase.close()

corpus = []

# pdb.set_trace()

for n, doc in enumerate(tqdm.tqdm(os.listdir(target_path))):
    Entdic = dict()
    EntityList = []
    removed_Entdic = dict()
    removed_EntityList = []
    Reldic = dict()
    REL_LABEL_unit = []
    Triggerdic = dict()


    max_length = int(config.get('makedata', 'MAX_SENT_LEN'))
    with open(target_path + doc, 'r', encoding='utf-8') as txtfile:
        text = txtfile.read()
    if text == '\n':
        continue
    #pdb.set_trace()

    # main prrocess -----------------------------------------------------------------------------------------------------------------------------------------------------------------

    spmed = sp.EncodeAsPieces(text)

    if len(spmed[0]) == 1:
        spmed[0] = '[CLS]'
    else:
        spmed[0] = spmed[0][1:]
        spmed.insert(0, '[CLS]')
    spmed.append('[SEP]')

    attention_mask      = [1] * len(spmed)
    target_spm          = [] 
    start_cnt = 0
    end_cnt   = 0

    for x, subword in enumerate(spmed[1:-1]):
        end_cnt = len(subword)
        target_spm.append((x, subword, start_cnt, start_cnt + end_cnt))
        start_cnt += end_cnt

    for i in range(max_length - len(spmed)):
        spmed.append('[PAD]')
        attention_mask.append(0)
    # pdb.set_trace()
    indx_tokens = [vocab[s] if s in vocab else vocab['<unk>'] for s in spmed]
    
    corpus.append(((n,doc), indx_tokens, attention_mask, spmed))

# (doc[0], indx_tokens, output_film_size1, output_film_size2, output_film_size3, output_film_size4, attention_mask, spmed, (n,doc,Entdic, Reldic))
n_Entdics_Reldics   = [a[0] for a in corpus]
word_input          = torch.LongTensor([a[1] for a in corpus]).to(device)
attention_mask      = torch.LongTensor([a[2] for a in corpus]).to(device)
sentencepieced      = [a[3] for a in corpus]

doc_correspnd_info_dict = {} #document毎にシンボリックな値をdocument名と辞書に変えるための辞書
n_doc = []
for unit in n_Entdics_Reldics:
    doc_correspnd_info_dict[unit[0]] = [(unit[1],unit[1][:-4]+'.ann'),{},{}]
    n_doc.append([unit[0]])

n_doc = torch.LongTensor(n_doc).to(device)
test_dataset = D.TensorDataset(n_doc, word_input, attention_mask)
test_loader  = D.DataLoader(test_dataset , batch_size=int(config.get('main', 'BATCH_SIZE_TEST' )), shuffle = strtobool(config.get('main', 'BATCH_SHUFFLE_TEST')))

print('finish', end='\n')
# pdb.set_trace()
print('Create Model...')
tokenizer = BERT_PRETRAINED_MODEL_JAPANESE(config, vocab).return_tokenizer()

model  = SPAN_CNN(config, vocab, REL_DIC).to(device)
# relation_model = BRAN(config, vocab, REL_DIC).to(device)
relation_model = RELATION(config, vocab, REL_DIC).to(device)
model.load_state_dict(torch.load(NER_model_save_path, map_location=device))
relation_model.load_state_dict(torch.load(RE_model_save_path, map_location=device))


# pretrained_BERT_model = bert_pretrained_model_japanese.return_model()
# tokenizer.convert_ids_to_tokens()というメソッドでindexを用語に直せる
print('finish', end='\n\n')

print('batch size is {0}'.format(batch))
print('Number of test is {0}'.format(len(word_input)))
print('learning rate is  {0}'.format(lr_str))

print('tensorboard writer logdir is {0}'.format(writer_log_dir))
print('brat logdir is {0}'.format(brat_log_dir))

start = time.time()
model.eval()
# pdb.set_trace()
epoch = 200

relation_flag = 1

data_unit_for_relation = []
predicts_relation  = []
answers_relation   = []
# pdb.set_trace()

print('\n start test ...\n')
model.eval()
predicted_tokens = []
n_docs = []
rel_ann_dict = defaultdict(list)

predicts_span1 = []
for_txt_span1  = []

predicts_span2 = []
for_txt_span2  = []

predicts_span3 = []
for_txt_span3  = []

predicts_span4 = []
for_txt_span4  = []

answers_token_preds_golds1 = []
answers_token_preds_golds2 = []
answers_token_preds_golds3 = []
answers_token_preds_golds4 = []
with torch.no_grad():
    for i, [n_doc, words, attention_mask] in enumerate(tqdm.tqdm(test_loader)):
        model.eval()
        batch_size = words.shape[0]

        # pdb.set_trace()
        pred_pair_unit = []
        model.zero_grad()
        logits_span1, logits_span2, logits_span3, logits_span4 = model(words, attention_mask, relation_flag, pred_pair_unit)
        predicts_span1.append(torch.max(logits_span1, 1)[1])
        predicts_span2.append(torch.max(logits_span2, 1)[1])
        predicts_span3.append(torch.max(logits_span3, 1)[1])
        predicts_span4.append(torch.max(logits_span4, 1)[1])

        ### text 化するための処理 batch内の全てのデータを一つにつなげてからpred2annに渡す ###########################################
        sentences = [a for a in words]
        for b_num in range(0, words.size(0)): #b_num番目のbatchを処理
            one_predicted_tokens = []
            one_for_txt_span1 = [] 
            one_for_txt_span2 = []
            one_for_txt_span3 = []
            one_for_txt_span4 = []
            for i_num in range(0, words.size(1)):
                predicted_token = tokenizer.convert_ids_to_tokens(sentences[b_num][i_num].item())
                one_predicted_tokens.append(predicted_token)
                one_for_txt_span1.append(torch.max(logits_span1, 1)[1][b_num][i_num].item())
                one_for_txt_span2.append(torch.max(logits_span2, 1)[1][b_num][i_num].item())
                one_for_txt_span3.append(torch.max(logits_span3, 1)[1][b_num][i_num].item())
                one_for_txt_span4.append(torch.max(logits_span4, 1)[1][b_num][i_num].item())
            predicted_tokens.append(one_predicted_tokens)
            for_txt_span1.append(one_for_txt_span1)
            for_txt_span2.append(one_for_txt_span2)
            for_txt_span3.append(one_for_txt_span3)
            for_txt_span4.append(one_for_txt_span4)
            n_docs.append(n_doc[b_num][0].item())

        if relation_flag == True: # Relationの予測
            # pdb.set_trace()
            s1,s2,s3,s4= nest_entity_process.nest_square_cut(predicts_span1,predicts_span2,predicts_span3,predicts_span4)
            pred_pairs_list_per_batch = PAIRS_MODULE(s1,s2,s3,s4).MAKE_PAIR()
            # REのDatasetを作る = 予測したペアに対してラベルを振る
            for number_in_minibatch, unique_number in enumerate(n_doc):
                # pdb.set_trace()
                pred_pairs = pred_pairs_list_per_batch[number_in_minibatch]
                for pred_pair_unit in pred_pairs:
                    relation_entity_all_label = (unique_number, words[number_in_minibatch].cpu().numpy(), attention_mask[number_in_minibatch].cpu().numpy(), pred_pair_unit)
                    data_unit_for_relation.append(relation_entity_all_label)
            # pdb.set_trace()

    if relation_flag == True:
        # pdb.set_trace()
        print("make Relation data ... ")
        unique_x             = torch.LongTensor([a[0] for a in data_unit_for_relation]).to(device)
        rel_word_x           = torch.LongTensor([a[1] for a in data_unit_for_relation]).to(device)
        rel_attention_mask   = torch.LongTensor([a[2] for a in data_unit_for_relation]).to(device)
        rel_pred_x           = torch.LongTensor([a[3] for a in data_unit_for_relation]).to(device)

        reldataset = D.TensorDataset(unique_x, rel_word_x, rel_attention_mask, rel_pred_x)
        rel_test_loader = D.DataLoader(reldataset, batch_size = int(config.get('main', 'REL_BATCH_SIZE_TRAIN')), shuffle = strtobool(config.get('main', 'BATCH_SHUFFLE_TRAIN')))
        print("finish")

        relation_model.eval()

        for r, [unique_x, rel_word_x, rel_attention_mask, rel_pred_x] in enumerate(progressbar(rel_test_loader)):
            relation_model.zero_grad()
            relation_logit = relation_model(rel_word_x, rel_attention_mask, rel_pred_x,0)
            predicts_relation.append(torch.max(relation_logit, 1)[1])
            for j, uni_x in enumerate(unique_x):
                rel_ann_dict[uni_x.item()].append((predicts_relation[0][j].item(),(rel_pred_x[j].to('cpu').numpy().tolist())))
            

    print('predictions -> annotations\n')
    pred_rel2ann.pred_rel2ann(epoch+1, brat_log_dir, doc_correspnd_info_dict, n_docs, predicted_tokens, for_txt_span1, for_txt_span2, for_txt_span3, for_txt_span4, rel_ann_dict, REL_DIC)

