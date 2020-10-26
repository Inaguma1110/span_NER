import sys
import argparse
import pdb
import tqdm, contextlib
import time
import shelve
import configparser
import datetime
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

from model.BRAN import JapaneseBertPretrainedModel, PairsModule, MyModel
from util import pred_rel2ann,pred2text,nest_entity_process

print('\nCreate Environment...\n')

## reading config file #############################################################
config = configparser.ConfigParser()
config.read('../machine_BRAN.conf')
dt_now = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

# dataname          = config.get('dataname', 'REL')  ### SPAN or SPAN_LARGE_NEST
dataname = config.get('dataname', "REL_DIVIDED_TEST")
max_sent_len      = int(config.get('makedata', 'MAX_SENT_LEN'))
network_structure = config.get('CNNs', 'NETWORK_STRUCTURE')
weight            = config.get('CNNs', 'WEIGHT')
batch             = config.get('main', 'BATCH_SIZE_TRAIN')
lr_str            = config.get('main', 'LEARNING_RATE')
attention_mask_is = config.get('CNNs', 'ATTENTION_MASK_IS')
TARGET_only_LARGE_NEST_flag = strtobool(config.get('main','NEST'))
NER_model_save_path = config.get('model path', 'NERmodel')
RE_model_save_path = config.get('model path', 'REmodel')

####################################################################################

## writer setting ##################################################################
# writer_log_dir ='../../data/TensorboardGraph/span_Joint/Test/'+dt_now+'batch_size_{0}/learning_rate_{1}/network_{2}_{3}/0_logit_weight_{4}/'.format(batch,lr_str,network_structure,attention_mask_is,weight)
writer_log_dir ='../../data/TensorboardGraph/for_jtekt'

# brat_log_dir   = '../../brat/brat-v1.3_Crunchy_Frog/data/model_preds/for_0826/span_Joint/Test/'+dt_now+'batch_size_{0}/learning_rate_{1}/network_{2}_{3}/0_logit_weight_{4}/'.format(batch,lr_str,network_structure,attention_mask_is,weight)
brat_log_dir   = '../../brat/brat-v1.3_Crunchy_Frog/data/model_preds/for_0826/'
model_writer_log_dir ='../../data/model/'
writer = tb.SummaryWriter(logdir = writer_log_dir)
# model_writer = SummaryWriter()
####################################################################################

np.random.seed(1)
torch.manual_seed(1)
# pdb.set_trace()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('finish', end='\n')
print('\nCreate data...')
database = shelve.open(config.get('path', 'SHELVE_PATH_TEST'))
vocab, REL_DIC, corpus, filename_lst,REL_LABEL_DICT = database[dataname] 
database.close()

# RELdatabase = shelve.open(config.get('path', 'REL_DIC_PATH'))
# REL_DIC = RELdatabase['REL_DIC']
# RELdatabase.close()

# (doc[0], indx_tokens, output_film_size1, output_film_size2, output_film_size3, output_film_size4, attention_mask, spmed, (n,doc,Entdic, Reldic))
n_Entdics_Reldics   = [a[0] for a in corpus]
word_input          = torch.LongTensor([a[1] for a in corpus]).to(device)
y_span_size_1       = torch.LongTensor([a[2] for a in corpus]).to(device) 
y_span_size_2       = torch.LongTensor([a[3] for a in corpus]).to(device)
y_span_size_3       = torch.LongTensor([a[4] for a in corpus]).to(device)
y_span_size_4       = torch.LongTensor([a[5] for a in corpus]).to(device)
attention_mask      = torch.LongTensor([a[6] for a in corpus]).to(device)
sentencepieced      = [a[7] for a in corpus]

doc_correspnd_info_dict = {} #document毎にシンボリックな値をdocument名と辞書に変えるための辞書
n_doc = []
for unit in n_Entdics_Reldics:
    doc_correspnd_info_dict[unit[0]] = unit[1:]
    n_doc.append([unit[0]])
n_doc = torch.LongTensor(n_doc).to(device)

test_dataset = D.TensorDataset(n_doc, word_input, attention_mask, y_span_size_1, y_span_size_2, y_span_size_3, y_span_size_4)
test_loader  = D.DataLoader(test_dataset , batch_size=int(config.get('main', 'BATCH_SIZE_TEST' )), shuffle = strtobool(config.get('main', 'BATCH_SHUFFLE_TEST')))

print('finish', end='\n')
# pdb.set_trace()
print('Create Model...')
tokenizer = JapaneseBertPretrainedModel(config, vocab).return_tokenizer()

NER_Model = MyModel(config, vocab, REL_DIC, REL_LABEL_DICT,doc_correspnd_info_dict).to(device)
# relation_model = BRAN(config, vocab, REL_DIC).to(device)
relation_model = MyModel(config, vocab, REL_DIC, REL_LABEL_DICT,doc_correspnd_info_dict).to(device)
NER_Model.load_state_dict(torch.load(NER_model_save_path, map_location=device))
relation_model.load_state_dict(torch.load(RE_model_save_path, map_location=device))


# pretrained_BERT_model = bert_pretrained_model_japanese.return_model()
# tokenizer.convert_ids_to_tokens()というメソッドでindexを用語に直せる
print('finish', end='\n\n')

# Loss関数の定義
weights = [float(s) for s in config.get('CNNs', 'LOSS_WEIGHT').split(',')]
class_weights = torch.FloatTensor(weights).to(device)
loss_function_span1 = nn.CrossEntropyLoss(weight=class_weights)
loss_function_span2 = nn.CrossEntropyLoss(weight=class_weights)
loss_function_span3 = nn.CrossEntropyLoss(weight=class_weights)
loss_function_span4 = nn.CrossEntropyLoss(weight=class_weights)

loss_function_relation = nn.CrossEntropyLoss()


print('target data is ' + dataname, end = '\n\n')
print('batch size is {0}'.format(batch))
print('Number of test is {0}'.format(len(word_input)))
print('learning rate is  {0}'.format(lr_str))
print('network strucuture is {0}'.format(network_structure))
NER_RE_switch = config.get('main', 'NER_RE_switch') # NER or RE or Joint

down_sampling_switch = 0
Relation_gold_learning_switch = 1

print('tensorboard writer logdir is {0}'.format(writer_log_dir))
print('brat logdir is {0}'.format(brat_log_dir))

start = time.time()

assert NER_RE_switch in ["NER", "RE", "Joint"]
do_re = NER_RE_switch in ["RE", "Joint"]
do_ner = NER_RE_switch in ["NER", "Joint"]


# pdb.set_trace()
epoch = 200
sum_loss = 0
sum_loss_relation = 0.0
sum_loss_span1 = 0.0
sum_loss_span2 = 0.0
sum_loss_span3 = 0.0
sum_loss_span4 = 0.0
span1_max_f = -1
span2_max_f = -1
span3_max_f = -1
span4_max_f = -1

relation_flag = 0

NER_max_scores = [span1_max_f, span2_max_f, span3_max_f, span4_max_f]

data_unit_for_relation = []

predicts_relation  = []
answers_relation   = []
# pdb.set_trace()

print('\n start test ...\n')

NER_Model.eval()
relation_model.eval()
predicted_tokens = []
n_docs = []
rel_ann_dict = defaultdict(list)

predicts_span1 = []
answers_span1  = []
for_txt_span1  = []

predicts_span2 = []
answers_span2  = []
for_txt_span2  = []

predicts_span3 = []
answers_span3  = []
for_txt_span3  = []

predicts_span4 = []
answers_span4  = []
for_txt_span4  = []

answers_token_preds_golds1 = []
answers_token_preds_golds2 = []
answers_token_preds_golds3 = []
answers_token_preds_golds4 = []
with torch.no_grad():
    for i, [
            n_doc, words, attention_mask, y_span_size_1, y_span_size_2,
            y_span_size_3, y_span_size_4
    ] in enumerate(tqdm.tqdm(test_loader)):
        # pdb.set_trace()
        NER_Model.eval()
        batch_size = words.shape[0]
        # pdb.set_trace()
        NER_Model.zero_grad()

        if do_ner:
            relation_flag == False
            logits_span1, logits_span2, logits_span3, logits_span4 = NER_Model(
                n_doc, words, attention_mask, relation_flag, NER_RE_switch,
                down_sampling_switch, y_span_size_1, y_span_size_2, y_span_size_3,
                y_span_size_4, Relation_gold_learning_switch)
            loss_span1 = loss_function_span1(logits_span1, y_span_size_1)
            loss_span2 = loss_function_span2(logits_span2, y_span_size_2)
            loss_span3 = loss_function_span3(logits_span3, y_span_size_3)
            loss_span4 = loss_function_span4(logits_span4, y_span_size_4)
            loss = loss_span1 + loss_span2 + loss_span3 + loss_span4

            sum_loss_span1 += float(loss_span1) * batch_size
            sum_loss_span2 += float(loss_span2) * batch_size
            sum_loss_span3 += float(loss_span3) * batch_size
            sum_loss_span4 += float(loss_span4) * batch_size
            sum_loss += float(loss) * batch_size

            predicts_span1.append(torch.max(logits_span1, 1)[1])
            predicts_span2.append(torch.max(logits_span2, 1)[1])
            predicts_span3.append(torch.max(logits_span3, 1)[1])
            predicts_span4.append(torch.max(logits_span4, 1)[1])

            answers_span1.append(y_span_size_1)
            answers_span2.append(y_span_size_2)
            answers_span3.append(y_span_size_3)
            answers_span4.append(y_span_size_4)

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


        if do_re:
            relation_flag = True
            relation_logit, rel_y = relation_model(n_doc, words, attention_mask,
                                          relation_flag, NER_RE_switch,
                                          down_sampling_switch, y_span_size_1,
                                          y_span_size_2, y_span_size_3,
                                          y_span_size_4,
                                          Relation_gold_learning_switch)
            loss_relation = loss_function_relation(relation_logit, rel_y)
            sum_loss_relation += float(loss_relation) * batch_size
            predicts_relation.append(torch.max(relation_logit, 1)[1])
            answers_relation.append(rel_y)
            # for j, uni_x in enumerate(n_doc):
            #     rel_ann_dict[uni_x.item()].append((predicts_relation[0][j].item(),(rel_pred_x[j].to('cpu').numpy().tolist())))
            relation_flag = False

pred_rel2ann.pred_rel2ann(epoch+1, brat_log_dir, doc_correspnd_info_dict, n_docs, predicted_tokens, for_txt_span1, for_txt_span2, for_txt_span3, for_txt_span4, rel_ann_dict, REL_DIC)



if do_ner:
    print("\n")
    print("Develop Loss Named Entity Recognition ...")
    print("Span1\tSpan2\tSpan3\tSpan4\tSum_Loss")
    print("{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}".format(
        sum_loss_span1, sum_loss_span2, sum_loss_span3, sum_loss_span4,
        sum_loss),
        end="\n\n")

    print('calculate span score ...')
    preds_span1 = torch.cat(predicts_span1, 0).view(-1, 1).squeeze().cpu().numpy()
    golds_span1 = torch.cat(answers_span1, 0).view(-1, 1).squeeze().cpu().numpy()

    preds_span2 = torch.cat(predicts_span2, 0).view(-1, 1).squeeze().cpu().numpy()
    golds_span2 = torch.cat(answers_span2, 0).view(-1, 1).squeeze().cpu().numpy()

    preds_span3 = torch.cat(predicts_span3, 0).view(-1, 1).squeeze().cpu().numpy()
    golds_span3 = torch.cat(answers_span3, 0).view(-1, 1).squeeze().cpu().numpy()

    preds_span4 = torch.cat(predicts_span4, 0).view(-1, 1).squeeze().cpu().numpy()
    golds_span4 = torch.cat(answers_span4, 0).view(-1, 1).squeeze().cpu().numpy()

    ######## nestの外部を評価するために内部の予測を0にしてから評価するための処理 ######
    if TARGET_only_LARGE_NEST_flag:
        pdb.set_trace()
        preds_span1, preds_span2, preds_span3, preds_span4 = nest_entity_process.nest_square_cut_for_eval(
            preds_span1, preds_span2, preds_span3, preds_span4)
        golds_span1, golds_span2, golds_span3, golds_span4 = nest_entity_process.nest_square_cut_for_eval(
            golds_span1, golds_span2, golds_span3, golds_span4)
    #########################################################################

    micro_p_r_f_span1 = precision_recall_fscore_support(golds_span1,preds_span1,labels=[1],average='micro')
    micro_p_r_f_span2 = precision_recall_fscore_support(golds_span2,preds_span2,labels=[1],average='micro')
    micro_p_r_f_span3 = precision_recall_fscore_support(golds_span3,preds_span3,labels=[1],average='micro')
    micro_p_r_f_span4 = precision_recall_fscore_support(golds_span4,preds_span4,labels=[1],average='micro')

    print('span1 micro p/r/F score is ' + str(micro_p_r_f_span1))
    print('span2 micro p/r/F score is ' + str(micro_p_r_f_span2))
    print('span3 micro p/r/F score is ' + str(micro_p_r_f_span3))
    print('span4 micro p/r/F score is ' + str(micro_p_r_f_span4))
    print()

    if span1_max_f < micro_p_r_f_span1[2]:
        span1_max_scores = (epoch, micro_p_r_f_span1)
        span1_max_f = micro_p_r_f_span1[2]
        NER_max_scores[0] = span1_max_scores

    if span2_max_f < micro_p_r_f_span2[2]:
        span2_max_scores = (epoch, micro_p_r_f_span2)
        span2_max_f = micro_p_r_f_span2[2]
        NER_max_scores[1] = span2_max_scores

    if span3_max_f < micro_p_r_f_span3[2]:
        span3_max_scores = (epoch, micro_p_r_f_span3)
        span3_max_f = micro_p_r_f_span3[2]
        NER_max_scores[2] = span3_max_scores

    if span4_max_f < micro_p_r_f_span4[2]:
        span4_max_scores = (epoch, micro_p_r_f_span4)
        span4_max_f = micro_p_r_f_span4[2]
        NER_max_scores[3] = span4_max_scores

    average_score = sum(
        list([a for a in zip(micro_p_r_f_span1, micro_p_r_f_span2,micro_p_r_f_span3, micro_p_r_f_span4)][2])) / 4
    print('span average score is {0} epoch is {1}'.format(average_score, epoch+1),end='\n\n\n')


    writer.add_scalar('span1_devel/loss_span1/epoch', sum_loss_span1, epoch)
    writer.add_scalar('span1_devel/micro_precision/epoch',micro_p_r_f_span1[0], epoch)
    writer.add_scalar('span1_devel/micro_recall/epoch', micro_p_r_f_span1[1],epoch)
    writer.add_scalar('span1_devel/micro_f-measure/epoch',micro_p_r_f_span1[2], epoch)

    writer.add_scalar('span2_devel/loss_span2/epoch', sum_loss_span2, epoch)
    writer.add_scalar('span2_devel/micro_precision/epoch',micro_p_r_f_span2[0], epoch)
    writer.add_scalar('span2_devel/micro_recall/epoch', micro_p_r_f_span2[1],epoch)
    writer.add_scalar('span2_devel/micro_f-measure/epoch',micro_p_r_f_span2[2], epoch)

    writer.add_scalar('span3_devel/loss_span3/epoch', sum_loss_span3, epoch)
    writer.add_scalar('span3_devel/micro_precision/epoch',micro_p_r_f_span3[0], epoch)
    writer.add_scalar('span3_devel/micro_recall/epoch', micro_p_r_f_span3[1],epoch)
    writer.add_scalar('span3_devel/micro_f-measure/epoch',micro_p_r_f_span3[2], epoch)

    writer.add_scalar('span4_devel/loss_span4/epoch', sum_loss_span4, epoch)
    writer.add_scalar('span4_devel/micro_precision/epoch',micro_p_r_f_span4[0], epoch)
    writer.add_scalar('span4_devel/micro_recall/epoch', micro_p_r_f_span4[1],epoch)
    writer.add_scalar('span4_devel/micro_f-measure/epoch',micro_p_r_f_span4[2], epoch)

if do_re:
    # pdb.set_trace()
    print("Develop Loss Relation Extraction ...")
    print("Relation Loss")
    print("{0:.3f}".format(sum_loss_relation), end='\n\n')
    print("calculate relation score ...")
    golds_relation = torch.cat(answers_relation, 0).cpu().numpy()
    preds_relation = torch.cat(predicts_relation, 0).cpu().numpy()

    micro_p_r_f_relation = precision_recall_fscore_support(golds_relation,preds_relation,labels=[1,2,3,4,5],average='micro')
    print('relation micro p/r/F score is ' + str(micro_p_r_f_relation),end='\n\n\n\n')



    writer.add_scalar('relation/loss/epoch', sum_loss_relation, epoch)
    writer.add_scalar('relation/micro_precision/epoch',
                        micro_p_r_f_relation[0], epoch)
    writer.add_scalar('relation/micro_recall/epoch',
                        micro_p_r_f_relation[1], epoch)
    writer.add_scalar('relation/micro_f-measure/epoch',
                        micro_p_r_f_relation[2], epoch)
    
    