import sys
import argparse
import datetime
import pdb
import tqdm
import contextlib
import time
import random
import shelve
import configparser
from progressbar import progressbar
from collections import defaultdict
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

from model.BRAN import JapaneseBertPretrainedModel, MyModel, PairsModule
from util import pred_rel2ann, pred2text, nest_entity_process


np.random.seed(1)
torch.manual_seed(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None

def joint_switch_controll(epoch):
    global NER_RE_switch, desc_tr, desc_dev, Relation_gold_learning_switch
    # pdb.set_trace()
    # if epoch == 20:
    #     pdb.set_trace()
    if epoch == 300:
        # if epoch == int(config.get('main', 'START_RELATION_EPOCH')):
        print(str("#" * 80) + '  Start Joint Learning  ' + str("#" * 90), end='\n')
        NER_RE_switch = "Joint"
        desc_tr = "Joint Learning train"
        desc_dev = "Joint Learning devel"
    if epoch == 500:
        # if epoch == int(config.get('main', 'START_RELATION_EPOCH')):
        print(str("#" * 80) + '  Start Joint Learning from NERdata  ' + str("#" * 90), end='\n')
        Relation_gold_learning_switch = 1
        
    if NER_RE_switch == "NER":
        print("NER Mode")

    if NER_RE_switch == "RE":
        print("RE Mode")
        if Relation_gold_learning_switch:
            print("Relation learned by Gold Entity Data")
        elif not Relation_gold_learning_switch:
            print("Relation learned by Extracted from NER data")

    if NER_RE_switch == "Joint":
        print("NER_RE Joint Mode")
        if Relation_gold_learning_switch:
            print("Relation learned by Gold Entity Data")
        elif not Relation_gold_learning_switch:
            print("Relation learned by Extracted from NER data")



dt_now = datetime.datetime.now(datetime.timezone(
    datetime.timedelta(hours=9))).strftime('%Y-%m-%d_%H:%M:%S')
print(dt_now)

# pdb.set_trace()
config = configparser.ConfigParser()
config.read('../machine_BRAN.conf')
max_sent_len = int(config.get('makedata', 'MAX_SENT_LEN'))

parser = argparse.ArgumentParser()
parser.add_argument('--structure',default = "NEST_INNER", help='NEST_INNER -> All nest entity\nNEST_OUTER -> only out nest entity')
parser.add_argument('--target_type',default = "Relation", help='All        -> all sentence   \nRelation   -> only relation sentence')
parser.add_argument('--span_size', default = 4, help = 'Choose span size \ndefault = 4')
parser.add_argument('--lr', default=1e-4, help='Set learning rate \ndefault = 1e-4')
parser.add_argument('--train_batch_size', default = 32, help = 'Set train batch size\n default = 32')
parser.add_argument('--target_only_large_nest_flag', default = False, help = 'Eval only NEST OUTER\ndefault = False')
parser.add_argument('--init_Joint_switch', default = 'Joint', help = 'Set mode of init training mode \ndefault = Joint')
parser.add_argument('--model_save', action='store_true')


data_structure_type = parser.parse_args().structure
target_data_type = parser.parse_args().target_type
span_size = parser.parse_args().span_size
lr_str = parser.parse_args().lr
train_batch = parser.parse_args().train_batch_size
TARGET_only_LARGE_NEST_flag = parser.parse_args().target_only_large_nest_flag
NER_RE_switch = parser.parse_args().init_Joint_switch
model_save = parser.parse_args().model_save

if model_save:
    print('\nModel save Mode\n')
    NER_model_save_path = config.get('model path', 'NERmodel')
    RE_model_save_path = config.get('model path', 'REmodel')

if not model_save:
    print('\nTrain Mode\n')

if target_data_type == "All":
    dataname = config.get('dataname', 'N_REL_DIVIDED_TRAIN_DEVEL')
if target_data_type == "Relation":
    dataname = config.get('dataname', 'N_REL_DIVIDED_TRAIN_DEVEL_SHORT')



writer_log_dir = '../../data/TensorboardGraph/span_Joint_Sep/'+dt_now + \
    'Jointmodel_relation/{0}/batch_size_{1}/learning_rate_{2}'.format(
        NER_RE_switch, train_batch, lr_str)
brat_log_dir = '../../brat/brat-v1.3_Crunchy_Frog/data/model_preds/span_NER_RE'+dt_now + \
    'Jointmodel_relation/{0}/batch_size_{1}/learning_rate_{2}'.format(
        NER_RE_switch, train_batch, lr_str)
model_writer_log_dir = '../../data/model/'
writer = tb.SummaryWriter(logdir=writer_log_dir)
# model_writer = SummaryWriter()



print('\nCreate data...')
database = shelve.open(config.get('path', 'SHELVE_PATH'))
vocab, REL_DIC, corpus, filename_lst, REL_LABEL_DICT = database[dataname]
database.close()

REL_database = shelve.open(config.get('path', 'REL_DIC_PATH'))
REL_DIC = REL_database['REL_DIC']
REL_database.close()
# pdb.set_trace()
# (doc[0], indx_tokens, output_films, tokinizerd_text, (n,doc,Entdic, Reldic))
n_Entdics_Reldics = [_[0] for _ in corpus]
word_input = torch.LongTensor([_[1] for _ in corpus]).to(device)
output_films = torch.LongTensor([_[2] for _ in corpus]).to(device)
attention_mask = torch.LongTensor([_[3] for _ in corpus]).to(device)
tokenized = [_[4] for _ in corpus]

doc_correspnd_info_dict = {}  # document毎にシンボリックな値をdocument名と辞書に変えるための辞書
n_doc = []
for unit in n_Entdics_Reldics:
    doc_correspnd_info_dict[unit[0]] = unit[1:]
    n_doc.append([unit[0]])
n_doc = torch.LongTensor(n_doc).to(device)

dataset = D.TensorDataset(n_doc, word_input, attention_mask, output_films)
train_size = int(0.9 * len(word_input))
devel_size = len(word_input) - train_size
train_dataset, devel_dataset = D.random_split(dataset,
                                              [train_size, devel_size])

train_loader = D.DataLoader(
    train_dataset,
    batch_size=int(config.get('main', 'BATCH_SIZE_TRAIN')),
    shuffle=strtobool(config.get('main', 'BATCH_SHUFFLE_TRAIN')))
devel_loader = D.DataLoader(
    devel_dataset,
    batch_size=int(config.get('main', 'BATCH_SIZE_DEVEL')),
    shuffle=strtobool(config.get('main', 'BATCH_SHUFFLE_DEVEL')))
print('finish', end='\n')


print('Create Model...')
Model = MyModel(span_size, config, vocab, REL_DIC, REL_LABEL_DICT,doc_correspnd_info_dict).to(device)
tokenizer = Model.bert_model.tokenizer

print('finish', end='\n\n')

# Loss関数の定義
loss_functions      =nn.ModuleList([nn.CrossEntropyLoss() for _ in range(span_size)])
loss_function_relation = nn.CrossEntropyLoss()
# Optimizerの設定
optimizer = optim.Adam(Model.parameters(), lr=float(lr_str),weight_decay=1e-3)
# rel_optimizer     = optim.Adam(relation_model.parameters(),lr=float(lr_str))
if strtobool(config.get('main', 'SCHEDULER')):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

start = time.time()
rel_ann_dict = defaultdict(list)



span1_max_f = -1
span2_max_f = -1
span3_max_f = -1
span4_max_f = -1
NER_max_scores = [span1_max_f, span2_max_f, span3_max_f, span4_max_f]
NER_max_scores = [-1 for _ in range(span_size)]


best_average_score = -1
best_re_score = -1

relation_flag = 0
Relation_gold_learning_switch = 1
down_sampling_switch = 0

switches = {}
switches['relation_flag'] = relation_flag
switches['NER_RE_switch'] = NER_RE_switch
switches['down_sampling_switch'] = down_sampling_switch
switches['Relation_gold_learning_switch'] = Relation_gold_learning_switch
switches['TARGET_only_LARGE_NEST_flag'] = TARGET_only_LARGE_NEST_flag
switches['model_save'] = model_save

def iterate(epoch, data_loader, Model, optimizer, switches, writer, is_training):
    relation_flag = switches['relation_flag']
    NER_RE_switch = switches['NER_RE_switch']
    down_sampling_switch = switches['down_sampling_switch']
    Relation_gold_learning_switch = switches['Relation_gold_learning_switch']
    
    if is_training:
        Model.train()
        desc = 'train_'
    else:
        Model.eval()
        desc = 'devel_'
    sum_loss = 0.0
    sum_losses = [0.0 for _ in range(span_size)]
    sum_loss_relation = 0.0
    predicts_spans = [[] for _ in range(span_size)]
    answers_spans  = [[] for _ in range(span_size)]

    preds = [[] for _ in range(span_size)]
    golds = [[] for _ in range(span_size)]

    micro_p_r_f_spans = [[] for _ in range(span_size)]

    predicts_relation = []
    answers_relation = []
    # if epoch == 40:
    #     pdb.set_trace()
    for i, [n_doc, words, attention_mask, y_spans] in enumerate(tqdm.tqdm(data_loader)):
        Model.zero_grad()
        batch_size = words.shape[0]
        all_loss = 0
        if do_ner:
            relation_flag = False
            logits_spans = Model(n_doc, words, attention_mask, relation_flag, NER_RE_switch, down_sampling_switch, y_spans, Relation_gold_learning_switch)
            for s_x in range(span_size):
                loss_span = loss_functions[s_x](logits_spans[s_x], y_spans.permute(1,0,2)[s_x])
                sum_losses[s_x] += loss_span
                all_loss += loss_span
                sum_loss += loss_span
                predicts_spans[s_x].append(torch.max(logits_spans[s_x], dim=1)[1])
                answers_spans[s_x].append(y_spans.permute(1,0,2)[s_x])

        if do_re:
            relation_flag = True
            relation_logit, rel_pred_x, rel_y = Model(n_doc, words, attention_mask, relation_flag, NER_RE_switch,down_sampling_switch, y_spans, Relation_gold_learning_switch)
            loss_relation = loss_function_relation(relation_logit, rel_y)
            sum_loss_relation += loss_relation
            all_loss += loss_relation
            relation_flag = False
            predicts_relation.append(torch.max(relation_logit, 1)[1])
            answers_relation.append(rel_y)

        if is_training:
            all_loss.backward(retain_graph=False)
            optimizer.step()
        else:
            scheduler.step(all_loss)
    # pdb.set_trace()


# ここから評価Part
    if do_ner:
        print("Loss is {0}".format(sum_loss))
        allloss_statement = 'NER/loss/' + desc + '/epoch' 
        for s_x in range(span_size):
            preds[s_x] = torch.cat(predicts_spans[s_x], 0).view(-1, 1).squeeze().cpu().numpy()
            golds[s_x] = torch.cat(answers_spans[s_x], 0).view(-1, 1).squeeze().cpu().numpy()
            micro_p_r_f_spans[s_x] = precision_recall_fscore_support(golds[s_x],preds[s_x],labels=[1],average='micro')

            p_r_f_statement = desc + 'span ' + str(s_x+1) + 'micro p/r/F score is'
            loss_statement = desc + 'span' + str(s_x+1) + '_/loss/epoch'
            precision_statement = desc + 'span' + str(s_x+1) + '_/micro_precision/epoch'
            recall_statement = desc + 'span' + str(s_x+1) + '_/micro_recall/epoch'
            f1_statement = desc + 'span' + str(s_x+1) + '_/micro_f1_value/epoch'
            statement_list = [loss_statement, precision_statement, recall_statement, f1_statement]
            for s, state in enumerate(statement_list):
                if s == 0:
                    print(desc + 'span {0} is {1}'.format(s_x+1, sum_losses[s_x]))
                    writer.add_scalar(state, sum_losses[s_x], epoch) #TODO
                if s > 0:
                    writer.add_scalar(state, micro_p_r_f_spans[s_x][s-1],epoch)
            print(p_r_f_statement + str(micro_p_r_f_spans[s_x]))
        writer.add_scalar(allloss_statement, all_loss, epoch)

    if do_re:
        golds_relation = torch.cat(answers_relation, 0).cpu().numpy()
        preds_relation = torch.cat(predicts_relation, 0).cpu().numpy()
        micro_p_r_f_relation = precision_recall_fscore_support(golds_relation,preds_relation,labels=[1,2,3,4,5],average='micro')
        print(desc + 'relation micro p/r/F score is ' + str(micro_p_r_f_relation),end='\n\n\n')
        print(desc + "Loss retlation is {0}".format(sum_loss_relation))
        print(desc + 'positive:',preds_relation.sum())

        loss_statement = desc + 'relation/loss/epoch'
        precision_statement = desc + 'relation/micro_precision/epoch'
        recall_statement = desc + 'relation/micro_recall/epoch'
        f1_statement = desc + 'relation/micro_f-measure/epoch'
        statement_list = [loss_statement, precision_statement, recall_statement, f1_statement]
        for s, state in enumerate(statement_list):
            if s == 0: 
                writer.add_scalar(state, loss_relation, epoch)
            if s > 0:
                writer.add_scalar(state, micro_p_r_f_relation[s-1],epoch)



# main 
# for epoch in range(int(config.get('main', 'N_EPOCH'))):
#     iterate(train_loader) 
#     Show train loss  #1epoch毎のTrainのLoss
#     iterate(devel_loader) 
#     Show dev loss  # 1epoch毎のDevのLoss



for epoch in range(int(config.get('main', 'N_EPOCH'))):
    print('Current Epoch:{}'.format(epoch + 1))

    joint_switch_controll

    assert NER_RE_switch in ["NER", "RE", "Joint"]
    do_re = NER_RE_switch in ["RE", "Joint"]
    do_ner = NER_RE_switch in ["NER", "Joint"]

    switches['down_sampling_switch'] = 1
    iterate(epoch, train_loader, Model, optimizer, switches, writer, is_training=True)

    switches['down_sampling_switch'] = 0
    iterate(epoch, devel_loader, Model, optimizer, switches, writer, is_training=False)










#     sum_loss = 0.0
#     sum_losses = [0.0 for _ in range(span_size)]
#     sum_loss_relation = 0.0
#     predicts_spans = [[] for _ in range(span_size)]
#     answers_spans  = [[] for _ in range(span_size)]

#     preds = [[] for _ in range(span_size)]
#     golds = [[] for _ in range(span_size)]

#     micro_p_r_f_spans = [[] for _ in range(span_size)]

#     predicts_relation = []
#     answers_relation = []

#     Model.train()

#     for i, [
#             n_doc, words, attention_mask, y_spans] in enumerate(tqdm.tqdm(train_loader, desc=desc_tr)):
#         Model.zero_grad()
#         batch_size = words.shape[0]
        
#         all_loss = 0

#         if do_ner:
#             relation_flag = False
#             logits_spans = Model(n_doc, words, attention_mask, relation_flag, NER_RE_switch,down_sampling_switch, y_spans, Relation_gold_learning_switch)

#             loss_span = [loss_functions[s_x](logits_spans.permute(2,0,1,3)[s_x], y_spans.permute(1,0,2)[s_x]) for s_x in range(span_size)]
#             for s_x in range(span_size):
                
#                 sum_losses[s_x] += loss_span[s_x]
#                 all_loss += loss_span[s_x]
#                 sum_loss += loss_span[s_x]
#                 predicts_spans[s_x].append(torch.max(logits_spans, dim=1)[1].permute(1,0,2)[s_x])
#                 answers_spans[s_x].append(y_spans.permute(1,0,2)[s_x])


#         if do_re:
#             relation_flag = True
#             relation_logit, rel_pred_x, rel_y = Model(n_doc, words, attention_mask, relation_flag, NER_RE_switch,down_sampling_switch, y_spans, Relation_gold_learning_switch)
            
#             loss_relation = loss_function_relation(relation_logit, rel_y)
#             sum_loss_relation += loss_relation
#             all_loss += loss_relation
            
#             relation_flag = False

#             predicts_relation.append(torch.max(relation_logit, 1)[1])
#             answers_relation.append(rel_y)
#             # print("{0} in data loader / len(rel_y) = {1}".format(i, len(rel_y)))

#         all_loss.backward(retain_graph=True)
#         optimizer.step()
#     # pdb.set_trace()

#     if do_ner:
#         print("Loss train is {0}".format(sum_loss))
#         writer.add_scalar('NER/loss/epoch', all_loss, epoch)
#         for s_x in range(span_size):
#             preds[s_x] = torch.cat(predicts_spans[s_x], 0).view(-1, 1).squeeze().cpu().numpy()
#             golds[s_x] = torch.cat(answers_spans[s_x], 0).view(-1, 1).squeeze().cpu().numpy()
#             micro_p_r_f_spans[s_x] = precision_recall_fscore_support(golds[s_x],preds[s_x],labels=[1],average='micro')

#             p_r_f_statement = 'train span ' + str(s_x+1) + 'micro p/r/F score is'
#             loss_statement = 'span' + str(s_x+1) + '_/loss/epoch'
#             precision_statement = 'span' + str(s_x+1) + '_/micro_precision/epoch'
#             recall_statement = 'span' + str(s_x+1) + '_/micro_recall/epoch'
#             f1_statement = 'span' + str(s_x+1) + '_/micro_f1_value/epoch'
#             statement_list = [loss_statement, precision_statement, recall_statement, f1_statement]
#             for s, state in enumerate(statement_list):
#                 if s == 0:
#                     writer.add_scalar(state, loss_span[s_x], epoch)
#                 if s > 0:
#                     writer.add_scalar(state, micro_p_r_f_spans[s_x][s-1],epoch)
#             print(p_r_f_statement + str(micro_p_r_f_spans[s_x]))

#     if do_re:
#         print("Loss retlation train is {0}".format(sum_loss_relation))
#         print('train positive:',preds_relation.sum())

#         golds_relation = torch.cat(answers_relation, 0).cpu().numpy()
#         preds_relation = torch.cat(predicts_relation, 0).cpu().numpy()
#         micro_p_r_f_relation = precision_recall_fscore_support(golds_relation,preds_relation,labels=[1,2,3,4,5],average='micro')
#         print('train relation micro p/r/F score is ' + str(micro_p_r_f_relation),end='\n\n\n')

#         loss_statement = 'relation/loss/epoch'
#         precision_statement = 'relation/micro_precision/epoch'
#         recall_statement = 'relation/micro_recall/epoch'
#         f1_statement = 'relation/micro_f-measure/epoch'
#         statement_list = [loss_statement, precision_statement, recall_statement, f1_statement]
#         for s, state in enumerate(statement_list):
#             if s == 0: 
#                 writer.add_scalar(state, loss_relation, epoch)
#             if s > 0:
#                 writer.add_scalar(state, micro_p_r_f_spans[s_x][s-1],epoch)



#     ########################################################## Develop process ###########################################################
#     down_sampling_switch = 0
#     # if NER_RE_switch == True:
#     #     pdb.set_trace()



#     sum_loss = 0.0
#     sum_losses = [0.0 for _ in range(span_size)]
#     sum_loss_relation = 0.0
#     predicts_spans = [[] for _ in range(span_size)]
#     answers_spans  = [[] for _ in range(span_size)]

#     preds = [[] for _ in range(span_size)]
#     golds = [[] for _ in range(span_size)]

#     micro_p_r_f_spans = [[] for _ in range(span_size)]

#     predicts_relation = []
#     answers_relation = []


#     for i, [
#             n_doc, words, attention_mask, y_span_size_1, y_span_size_2,
#             y_span_size_3, y_span_size_4
#     ] in enumerate(tqdm.tqdm(devel_loader, desc=desc_dev)):
#         # pdb.set_trace()
#         Model.eval()
#         batch_size = words.shape[0]
#         # pdb.set_trace()
#         Model.zero_grad()

#         if do_ner:
#             relation_flag == False
#             logits_span1, logits_span2, logits_span3, logits_span4 = Model(
#                 n_doc, words, attention_mask, relation_flag, NER_RE_switch,
#                 down_sampling_switch, y_span_size_1, y_span_size_2, y_span_size_3,
#                 y_span_size_4, Relation_gold_learning_switch)
#             loss_span1 = loss_function_span1(logits_span1, y_span_size_1)
#             loss_span2 = loss_function_span2(logits_span2, y_span_size_2)
#             loss_span3 = loss_function_span3(logits_span3, y_span_size_3)
#             loss_span4 = loss_function_span4(logits_span4, y_span_size_4)
#             loss = loss_span1 + loss_span2 + loss_span3 + loss_span4
#             if strtobool(config.get('main', 'SCHEDULER')):
#                 scheduler.step(loss)
#             sum_loss_span1 += float(loss_span1) * batch_size
#             sum_loss_span2 += float(loss_span2) * batch_size
#             sum_loss_span3 += float(loss_span3) * batch_size
#             sum_loss_span4 += float(loss_span4) * batch_size
#             sum_loss += float(loss) * batch_size

#             predicts_span1.append(torch.max(logits_span1, 1)[1])
#             predicts_span2.append(torch.max(logits_span2, 1)[1])
#             predicts_span3.append(torch.max(logits_span3, 1)[1])
#             predicts_span4.append(torch.max(logits_span4, 1)[1])

#             answers_span1.append(y_span_size_1)
#             answers_span2.append(y_span_size_2)
#             answers_span3.append(y_span_size_3)
#             answers_span4.append(y_span_size_4)

#         if do_re:
#             relation_flag = True
#             relation_logit, rel_pred_x, rel_y = Model(n_doc, words, attention_mask,
#                                           relation_flag, NER_RE_switch,
#                                           down_sampling_switch, y_span_size_1,
#                                           y_span_size_2, y_span_size_3,
#                                           y_span_size_4,
#                                           Relation_gold_learning_switch)
#             loss_relation = loss_function_relation(relation_logit, rel_y)
#             sum_loss_relation += float(loss_relation) * batch_size
#             if strtobool(config.get('main', 'SCHEDULER')):
#                 scheduler.step(loss_relation)
#             predicts_relation.append(torch.max(relation_logit, 1)[1])
#             answers_relation.append(rel_y)
#             relation_flag = False



#     if do_ner:
#         print("\n")
#         print("Develop Loss Named Entity Recognition ...")
#         print("Span1\tSpan2\tSpan3\tSpan4\tSum_Loss")
#         print("{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}".format(
#             sum_loss_span1, sum_loss_span2, sum_loss_span3, sum_loss_span4,
#             sum_loss),
#             end="\n\n") 

#         print('calculate span score ...')
#         preds_span1 = torch.cat(predicts_span1, 0).view(-1, 1).squeeze().cpu().numpy()
#         golds_span1 = torch.cat(answers_span1, 0).view(-1, 1).squeeze().cpu().numpy()

#         preds_span2 = torch.cat(predicts_span2, 0).view(-1, 1).squeeze().cpu().numpy()
#         golds_span2 = torch.cat(answers_span2, 0).view(-1, 1).squeeze().cpu().numpy()

#         preds_span3 = torch.cat(predicts_span3, 0).view(-1, 1).squeeze().cpu().numpy()
#         golds_span3 = torch.cat(answers_span3, 0).view(-1, 1).squeeze().cpu().numpy()

#         preds_span4 = torch.cat(predicts_span4, 0).view(-1, 1).squeeze().cpu().numpy()
#         golds_span4 = torch.cat(answers_span4, 0).view(-1, 1).squeeze().cpu().numpy()

#         ######## nestの外部を評価するために内部の予測を0にしてから評価するための処理 ######
#         if TARGET_only_LARGE_NEST_flag:
#             preds_span1, preds_span2, preds_span3, preds_span4 = nest_entity_process.nest_square_cut_for_eval(
#                 preds_span1, preds_span2, preds_span3, preds_span4)
#             golds_span1, golds_span2, golds_span3, golds_span4 = nest_entity_process.nest_square_cut_for_eval(
#                 golds_span1, golds_span2, golds_span3, golds_span4)
#         #########################################################################

#         micro_p_r_f_span1 = precision_recall_fscore_support(golds_span1,preds_span1,labels=[1],average='micro')
#         micro_p_r_f_span2 = precision_recall_fscore_support(golds_span2,preds_span2,labels=[1],average='micro')
#         micro_p_r_f_span3 = precision_recall_fscore_support(golds_span3,preds_span3,labels=[1],average='micro')
#         micro_p_r_f_span4 = precision_recall_fscore_support(golds_span4,preds_span4,labels=[1],average='micro')

#         print('span1 micro p/r/F score is ' + str(micro_p_r_f_span1))
#         print('span2 micro p/r/F score is ' + str(micro_p_r_f_span2))
#         print('span3 micro p/r/F score is ' + str(micro_p_r_f_span3))
#         print('span4 micro p/r/F score is ' + str(micro_p_r_f_span4))
#         print()

#         if span1_max_f < micro_p_r_f_span1[2]:
#             span1_max_scores = (epoch, micro_p_r_f_span1)
#             span1_max_f = micro_p_r_f_span1[2]
#             NER_max_scores[0] = span1_max_scores

#         if span2_max_f < micro_p_r_f_span2[2]:
#             span2_max_scores = (epoch, micro_p_r_f_span2)
#             span2_max_f = micro_p_r_f_span2[2]
#             NER_max_scores[1] = span2_max_scores

#         if span3_max_f < micro_p_r_f_span3[2]:
#             span3_max_scores = (epoch, micro_p_r_f_span3)
#             span3_max_f = micro_p_r_f_span3[2]
#             NER_max_scores[2] = span3_max_scores

#         if span4_max_f < micro_p_r_f_span4[2]:
#             span4_max_scores = (epoch, micro_p_r_f_span4)
#             span4_max_f = micro_p_r_f_span4[2]
#             NER_max_scores[3] = span4_max_scores

#         average_score = sum(
#             list([_ for _ in zip(micro_p_r_f_span1, micro_p_r_f_span2,micro_p_r_f_span3, micro_p_r_f_span4)][2])) / 4
#         print('span average score is {0} epoch is {1}'.format(average_score, epoch+1),end='\n\n\n')
#         if average_score > best_average_score:
#             best_average_score = average_score
#             if model_save:
#                 torch.save(Model.state_dict(), NER_model_save_path)

#         writer.add_scalar('span1_devel/loss_span1/epoch', sum_loss_span1, epoch)
#         writer.add_scalar('span1_devel/micro_precision/epoch',micro_p_r_f_span1[0], epoch)
#         writer.add_scalar('span1_devel/micro_recall/epoch', micro_p_r_f_span1[1],epoch)
#         writer.add_scalar('span1_devel/micro_f-measure/epoch',micro_p_r_f_span1[2], epoch)

#         writer.add_scalar('span2_devel/loss_span2/epoch', sum_loss_span2, epoch)
#         writer.add_scalar('span2_devel/micro_precision/epoch',micro_p_r_f_span2[0], epoch)
#         writer.add_scalar('span2_devel/micro_recall/epoch', micro_p_r_f_span2[1],epoch)
#         writer.add_scalar('span2_devel/micro_f-measure/epoch',micro_p_r_f_span2[2], epoch)

#         writer.add_scalar('span3_devel/loss_span3/epoch', sum_loss_span3, epoch)
#         writer.add_scalar('span3_devel/micro_precision/epoch',micro_p_r_f_span3[0], epoch)
#         writer.add_scalar('span3_devel/micro_recall/epoch', micro_p_r_f_span3[1],epoch)
#         writer.add_scalar('span3_devel/micro_f-measure/epoch',micro_p_r_f_span3[2], epoch)

#         writer.add_scalar('span4_devel/loss_span4/epoch', sum_loss_span4, epoch)
#         writer.add_scalar('span4_devel/micro_precision/epoch',micro_p_r_f_span4[0], epoch)
#         writer.add_scalar('span4_devel/micro_recall/epoch', micro_p_r_f_span4[1],epoch)
#         writer.add_scalar('span4_devel/micro_f-measure/epoch',micro_p_r_f_span4[2], epoch)

#     if do_re:
#         # pdb.set_trace()
#         print("Develop Loss Relation Extraction ...")
#         print("Relation Loss")
#         print("{0:.3f}".format(sum_loss_relation), end='\n\n')
#         print("calculate relation score ...")
#         golds_relation = torch.cat(answers_relation, 0).cpu().numpy()
#         preds_relation = torch.cat(predicts_relation, 0).cpu().numpy()

#         micro_p_r_f_relation = precision_recall_fscore_support(golds_relation,preds_relation,labels=[1,2,3,4,5],average='micro')
#         print('relation micro p/r/F score is ' + str(micro_p_r_f_relation),end='\n\n\n\n')

#         if best_re_score < micro_p_r_f_relation[2]:
#             best_re_score = micro_p_r_f_relation[2]
#             if model_save:
#                 torch.save(Model.state_dict(), RE_model_save_path)

#         writer.add_scalar('relation/loss/epoch', sum_loss_relation, epoch)
#         writer.add_scalar('relation/micro_precision/epoch',
#                           micro_p_r_f_relation[0], epoch)
#         writer.add_scalar('relation/micro_recall/epoch',
#                           micro_p_r_f_relation[1], epoch)
#         writer.add_scalar('relation/micro_f-measure/epoch',
#                           micro_p_r_f_relation[2], epoch)



# pdb.set_trace()
# dataname = config.get('dataname', "REL_DIVIDED_TEST_SHORT")

# database = shelve.open(config.get('path', 'SHELVE_PATH'))
# vocab, REL_DIC, corpus, filename_lst, REL_LABEL_DICT = database[dataname]
# database.close()

# n_Entdics_Reldics = [_[0] for _ in corpus]
# word_input = torch.LongTensor([_[1] for _ in corpus]).to(device)
# y_span_size_1 = torch.LongTensor([_[2] for _ in corpus]).to(device)
# y_span_size_2 = torch.LongTensor([_[3] for _ in corpus]).to(device)
# y_span_size_3 = torch.LongTensor([_[4] for _ in corpus]).to(device)
# y_span_size_4 = torch.LongTensor([_[5] for _ in corpus]).to(device)
# attention_mask = torch.LongTensor([_[6] for _ in corpus]).to(device)
# sentencepieced = [_[7] for _ in corpus]
# # vocab, REL_DIC, corpus, filename_lst, REL_LABEL_DICT = database[dataname]

# doc_correspnd_info_dict = {}  # document毎にシンボリックな値をdocument名と辞書に変えるための辞書
# n_doc = []
# n_docs = []
# predicted_tokens = []
# for_txt_span1  = []
# for_txt_span2  = []
# for_txt_span3  = []
# for_txt_span4  = []


# for unit in n_Entdics_Reldics:
#     doc_correspnd_info_dict[unit[0]] = unit[1:]
#     n_doc.append([unit[0]])
# n_doc = torch.LongTensor(n_doc).to(device)

# test_dataset = D.TensorDataset(n_doc, word_input, attention_mask, y_span_size_1, y_span_size_2, y_span_size_3, y_span_size_4)
# test_loader  = D.DataLoader(test_dataset , batch_size=int(config.get('main', 'BATCH_SIZE_TEST' )), shuffle = strtobool(config.get('main', 'BATCH_SHUFFLE_TEST')))

# for i, [
#         n_doc, words, attention_mask, y_span_size_1, y_span_size_2,
#         y_span_size_3, y_span_size_4
# ] in enumerate(tqdm.tqdm(test_loader, desc=desc_dev)):
#     # pdb.set_trace()
#     Model.eval()
#     batch_size = words.shape[0]
#     # pdb.set_trace()
#     Model.zero_grad()

#     if do_ner:
#         relation_flag == False
#         logits_span1, logits_span2, logits_span3, logits_span4 = Model(
#             n_doc, words, attention_mask, relation_flag, NER_RE_switch,
#             down_sampling_switch, y_span_size_1, y_span_size_2, y_span_size_3,
#             y_span_size_4, Relation_gold_learning_switch)
#         loss_span1 = loss_function_span1(logits_span1, y_span_size_1)
#         loss_span2 = loss_function_span2(logits_span2, y_span_size_2)
#         loss_span3 = loss_function_span3(logits_span3, y_span_size_3)
#         loss_span4 = loss_function_span4(logits_span4, y_span_size_4)
#         loss = loss_span1 + loss_span2 + loss_span3 + loss_span4
#         if strtobool(config.get('main', 'SCHEDULER')):
#             scheduler.step(loss)
#         sum_loss_span1 += float(loss_span1) * batch_size
#         sum_loss_span2 += float(loss_span2) * batch_size
#         sum_loss_span3 += float(loss_span3) * batch_size
#         sum_loss_span4 += float(loss_span4) * batch_size
#         sum_loss += float(loss) * batch_size

#         predicts_span1.append(torch.max(logits_span1, 1)[1])
#         predicts_span2.append(torch.max(logits_span2, 1)[1])
#         predicts_span3.append(torch.max(logits_span3, 1)[1])
#         predicts_span4.append(torch.max(logits_span4, 1)[1])

#         answers_span1.append(y_span_size_1)
#         answers_span2.append(y_span_size_2)
#         answers_span3.append(y_span_size_3)
#         answers_span4.append(y_span_size_4)

#         sentences = [_ for _ in words]
#         for b_num in range(0, words.size(0)): #b_num番目のbatchを処理
#             one_predicted_tokens = []
#             one_for_txt_span1 = [] 
#             one_for_txt_span2 = []
#             one_for_txt_span3 = []
#             one_for_txt_span4 = []
#             for i_num in range(0, words.size(1)):
#                 predicted_token = tokenizer.convert_ids_to_tokens(sentences[b_num][i_num].item())
#                 one_predicted_tokens.append(predicted_token)
#                 one_for_txt_span1.append(torch.max(logits_span1, 1)[1][b_num][i_num].item())
#                 one_for_txt_span2.append(torch.max(logits_span2, 1)[1][b_num][i_num].item())
#                 one_for_txt_span3.append(torch.max(logits_span3, 1)[1][b_num][i_num].item())
#                 one_for_txt_span4.append(torch.max(logits_span4, 1)[1][b_num][i_num].item())
#             predicted_tokens.append(one_predicted_tokens)
#             for_txt_span1.append(one_for_txt_span1)
#             for_txt_span2.append(one_for_txt_span2)
#             for_txt_span3.append(one_for_txt_span3)
#             for_txt_span4.append(one_for_txt_span4)
#             n_docs.append(n_doc[b_num][0].item())
#     if do_re:
#         relation_flag = True
#         relation_logit, rel_pred_x, rel_y = Model(n_doc, words, attention_mask,
#                                         relation_flag, NER_RE_switch,
#                                         down_sampling_switch, y_span_size_1,
#                                         y_span_size_2, y_span_size_3,
#                                         y_span_size_4,
#                                         Relation_gold_learning_switch)
#         loss_relation = loss_function_relation(relation_logit, rel_y)
#         sum_loss_relation += float(loss_relation) * batch_size
#         if strtobool(config.get('main', 'SCHEDULER')):
#             scheduler.step(loss_relation)
#         predicts_relation.append(torch.max(relation_logit, 1)[1])
#         answers_relation.append(rel_y)
#         relation_flag = False

#         for j, uni_x in enumerate(n_doc):
#             rel_ann_dict[uni_x.item()].append((predicts_relation[0][j].item(),(rel_pred_x[j].to('cpu').numpy().tolist())))



# if do_ner:
#     print("\n")
#     print("test Loss Named Entity Recognition ...")
#     print("Span1\tSpan2\tSpan3\tSpan4\tSum_Loss")
#     print("{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}".format(
#         sum_loss_span1, sum_loss_span2, sum_loss_span3, sum_loss_span4,
#         sum_loss),
#         end="\n\n") 

#     print('calculate span score ...')
#     preds_span1 = torch.cat(predicts_span1, 0).view(-1, 1).squeeze().cpu().numpy()
#     golds_span1 = torch.cat(answers_span1, 0).view(-1, 1).squeeze().cpu().numpy()

#     preds_span2 = torch.cat(predicts_span2, 0).view(-1, 1).squeeze().cpu().numpy()
#     golds_span2 = torch.cat(answers_span2, 0).view(-1, 1).squeeze().cpu().numpy()

#     preds_span3 = torch.cat(predicts_span3, 0).view(-1, 1).squeeze().cpu().numpy()
#     golds_span3 = torch.cat(answers_span3, 0).view(-1, 1).squeeze().cpu().numpy()

#     preds_span4 = torch.cat(predicts_span4, 0).view(-1, 1).squeeze().cpu().numpy()
#     golds_span4 = torch.cat(answers_span4, 0).view(-1, 1).squeeze().cpu().numpy()

#     ######## nestの外部を評価するために内部の予測を0にしてから評価するための処理 ######
#     if TARGET_only_LARGE_NEST_flag:
#         preds_span1, preds_span2, preds_span3, preds_span4 = nest_entity_process.nest_square_cut_for_eval(
#             preds_span1, preds_span2, preds_span3, preds_span4)
#         golds_span1, golds_span2, golds_span3, golds_span4 = nest_entity_process.nest_square_cut_for_eval(
#             golds_span1, golds_span2, golds_span3, golds_span4)
#     #########################################################################

#     micro_p_r_f_span1 = precision_recall_fscore_support(golds_span1,preds_span1,labels=[1],average='micro')
#     micro_p_r_f_span2 = precision_recall_fscore_support(golds_span2,preds_span2,labels=[1],average='micro')
#     micro_p_r_f_span3 = precision_recall_fscore_support(golds_span3,preds_span3,labels=[1],average='micro')
#     micro_p_r_f_span4 = precision_recall_fscore_support(golds_span4,preds_span4,labels=[1],average='micro')

#     print('span1 micro p/r/F score is ' + str(micro_p_r_f_span1))
#     print('span2 micro p/r/F score is ' + str(micro_p_r_f_span2))
#     print('span3 micro p/r/F score is ' + str(micro_p_r_f_span3))
#     print('span4 micro p/r/F score is ' + str(micro_p_r_f_span4))
#     print()

#     if span1_max_f < micro_p_r_f_span1[2]:
#         span1_max_scores = (epoch, micro_p_r_f_span1)
#         span1_max_f = micro_p_r_f_span1[2]
#         NER_max_scores[0] = span1_max_scores

#     if span2_max_f < micro_p_r_f_span2[2]:
#         span2_max_scores = (epoch, micro_p_r_f_span2)
#         span2_max_f = micro_p_r_f_span2[2]
#         NER_max_scores[1] = span2_max_scores

#     if span3_max_f < micro_p_r_f_span3[2]:
#         span3_max_scores = (epoch, micro_p_r_f_span3)
#         span3_max_f = micro_p_r_f_span3[2]
#         NER_max_scores[2] = span3_max_scores

#     if span4_max_f < micro_p_r_f_span4[2]:
#         span4_max_scores = (epoch, micro_p_r_f_span4)
#         span4_max_f = micro_p_r_f_span4[2]
#         NER_max_scores[3] = span4_max_scores

#     average_score = sum(
#         list([_ for _ in zip(micro_p_r_f_span1, micro_p_r_f_span2,micro_p_r_f_span3, micro_p_r_f_span4)][2])) / 4
#     print('span average score is {0} epoch is {1}'.format(average_score, epoch+1),end='\n\n\n')
#     if average_score > best_average_score:
#         best_average_score = average_score
#         if model_save:
#             torch.save(Model.state_dict(), NER_model_save_path)

#     writer.add_scalar('span1_test/loss_span1/epoch', sum_loss_span1, epoch)
#     writer.add_scalar('span1_test/micro_precision/epoch',micro_p_r_f_span1[0], epoch)
#     writer.add_scalar('span1_test/micro_recall/epoch', micro_p_r_f_span1[1],epoch)
#     writer.add_scalar('span1_test/micro_f-measure/epoch',micro_p_r_f_span1[2], epoch)

#     writer.add_scalar('span2_test/loss_span2/epoch', sum_loss_span2, epoch)
#     writer.add_scalar('span2_test/micro_precision/epoch',micro_p_r_f_span2[0], epoch)
#     writer.add_scalar('span2_test/micro_recall/epoch', micro_p_r_f_span2[1],epoch)
#     writer.add_scalar('span2_test/micro_f-measure/epoch',micro_p_r_f_span2[2], epoch)

#     writer.add_scalar('span3_test/loss_span3/epoch', sum_loss_span3, epoch)
#     writer.add_scalar('span3_test/micro_precision/epoch',micro_p_r_f_span3[0], epoch)
#     writer.add_scalar('span3_test/micro_recall/epoch', micro_p_r_f_span3[1],epoch)
#     writer.add_scalar('span3_test/micro_f-measure/epoch',micro_p_r_f_span3[2], epoch)

#     writer.add_scalar('span4_test/loss_span4/epoch', sum_loss_span4, epoch)
#     writer.add_scalar('span4_test/micro_precision/epoch',micro_p_r_f_span4[0], epoch)
#     writer.add_scalar('span4_test/micro_recall/epoch', micro_p_r_f_span4[1],epoch)
#     writer.add_scalar('span4_test/micro_f-measure/epoch',micro_p_r_f_span4[2], epoch)

    

# if do_re:
#     # pdb.set_trace()
#     print("test Loss Relation Extraction ...")
#     print("Relation Loss")
#     print("{0:.3f}".format(sum_loss_relation), end='\n\n')
#     print("calculate relation score ...")
#     golds_relation = torch.cat(answers_relation, 0).cpu().numpy()
#     preds_relation = torch.cat(predicts_relation, 0).cpu().numpy()

#     micro_p_r_f_relation = precision_recall_fscore_support(golds_relation,preds_relation,labels=[1,2,3,4,5],average='micro')
#     print('relation micro p/r/F score is ' + str(micro_p_r_f_relation),end='\n\n\n\n')


#     if best_re_score < micro_p_r_f_relation[2]:
#         best_re_score = micro_p_r_f_relation[2]
#         if model_save:
#             torch.save(Model.state_dict(), RE_model_save_path)

#     writer.add_scalar('relation/loss/epoch', sum_loss_relation, epoch)
#     writer.add_scalar('relation/micro_precision/epoch',
#                         micro_p_r_f_relation[0], epoch)
#     writer.add_scalar('relation/micro_recall/epoch',
#                         micro_p_r_f_relation[1], epoch)
#     writer.add_scalar('relation/micro_f-measure/epoch',
#                         micro_p_r_f_relation[2], epoch)

#     print('predictions -> annotations\n')
#     pred_rel2ann.pred_rel2ann(epoch+1, brat_log_dir, doc_correspnd_info_dict, n_docs, predicted_tokens, for_txt_span1, for_txt_span2, for_txt_span3, for_txt_span4, rel_ann_dict, REL_DIC)


