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

def print_setting():
    print()
    print('data_structure_type is {0}'.format(data_structure_type))
    print('target_data_type is {0}'.format(target_data_type))
    print('span_size is {0}'.format(span_size))
    print('lr_str is {0}'.format(lr_str))
    print('train_batch is {0}'.format(train_batch))
    print('scheduler_switch is {0}'.format(scheduler_switch))
    print('TARGET_only_LARGE_NEST_flag is {0}'.format(TARGET_only_LARGE_NEST_flag))
    print('model_save is {0}'.format(model_save))
    print('is_writer is {0}'.format(is_writer))
    print('NER_RE_switch is {0}'.format(NER_RE_switch))

def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None


def joint_switch_controll(epoch, switches):
    global desc_tr, desc_dev, Relation_gold_learning_switch

    if epoch == 101:
        # if epoch == int(config.get('main', 'START_RELATION_EPOCH')):
        print(str("#" * 80) + '  Start Joint Learning  ' + str("#" * 90), end='\n')
        switches['NER_RE_switch'] = "Joint"
        desc_tr = "Joint Learning train"
        desc_dev = "Joint Learning devel"
        switches['Relation_gold_learning_switch'] = 1
    
    
    if switches['NER_RE_switch'] == "NER":
        print("NER Mode")

    if switches['NER_RE_switch'] == "RE":
        print("RE Mode")
        if Relation_gold_learning_switch:
            print("Relation learned by Gold Entity Data")
        elif not Relation_gold_learning_switch:
            print("Relation learned by Extracted from NER data")

    if switches['NER_RE_switch'] == "Joint":
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

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--structure',default = "NEST_INNER", help='NEST_INNER -> All nest entity\nNEST_OUTER -> only out nest entity')
parser.add_argument('--target_type',default = "Relation", help='All -> all sentence   \nRelation   -> only relation sentence')
parser.add_argument('--span_size', default = 4, help = 'Choose span size (default : 4)')
parser.add_argument('--lr', default=1e-4, help='Set learning rate (default : 1e-4)')
parser.add_argument('--train_batch_size', default = 32, help = 'Set train batch size (default : 32)')
parser.add_argument('--is_scheduler',  action='store_true', help = 'Set optimize scheduler (default : False)')
parser.add_argument('--target_only_large_nest_flag', action='store_true' ,help = 'Eval only NEST OUTER (default : False)')
parser.add_argument('--model_save', action='store_true')
parser.add_argument('--is_writer', action='store_true', help = 'If want to record score log, add --is_writer')

parser.add_argument('--init_Joint_switch', default = 'Joint', help = 'Set mode of init training mode (default : Joint)')

data_structure_type = parser.parse_args().structure
target_data_type = parser.parse_args().target_type
span_size = int(parser.parse_args().span_size)
lr_str = parser.parse_args().lr
train_batch = parser.parse_args().train_batch_size
scheduler_switch = parser.parse_args().is_scheduler
TARGET_only_LARGE_NEST_flag = parser.parse_args().target_only_large_nest_flag
model_save = parser.parse_args().model_save
is_writer = parser.parse_args().is_writer

NER_RE_switch = parser.parse_args().init_Joint_switch

print_setting()


if not model_save:
    print('\nTrain Mode\n')

if target_data_type == "All":
    dataname = str(span_size) + config.get('dataname', 'N_REL_DIVIDED_TEST')
if target_data_type == "Relation":
    dataname = str(span_size) + config.get('dataname', 'N_REL_DIVIDED_TEST_SHORT')




brat_log_dir = '../../brat/brat-v1.3_Crunchy_Frog/data/model_preds/span_NER_RE'+dt_now + \
    'Jointmodel_relation/{0}/batch_size_{1}/learning_rate_{2}'.format(
        NER_RE_switch, train_batch, lr_str)
model_writer_log_dir = '../../data/model/'
if is_writer:
    writer_log_dir = '../../data/TensorboardGraph/span_Joint_Sep_ex/'+dt_now + \
        'Jointmodel_relation/{0}/span_size_{1}_scheduler_is_{2}/learning_rate_{3}'.format(
        NER_RE_switch, span_size, scheduler_switch, lr_str)
    writer = tb.SummaryWriter(logdir=writer_log_dir)
else:
    writer = 0
joint_model_path = config.get('model path', 'jointmodel')
# model_writer = SummaryWriter()



print('\nCreate data...')
database = shelve.open(config.get('path', 'SHELVE_PATH'))
vocab, rel_dic, corpus, filename_lst, REL_LABEL_DICT = database[dataname]
database.close()

REL_database = shelve.open(config.get('path', 'REL_DIC_PATH'))
rel_dic = REL_database['rel_dic']
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
print(len(word_input))
dataset = D.TensorDataset(n_doc, word_input, attention_mask, output_films)
test_loader = D.DataLoader(dataset,batch_size = 4,shuffle=False)

print('finish', end='\n')


print('Create Model...')
Model = MyModel(span_size, config, vocab, rel_dic, REL_LABEL_DICT,doc_correspnd_info_dict).to(device)
Model.load_state_dict(torch.load(joint_model_path))
tokenizer = Model.bert_model.tokenizer
print('finish', end='\n\n')

# Loss関数の定義
loss_functions      =nn.ModuleList([nn.CrossEntropyLoss() for _ in range(span_size)])
loss_function_relation = nn.CrossEntropyLoss()
# Optimizerの設定
optimizer = optim.Adam(Model.parameters(), lr=float(lr_str),weight_decay=1e-3)
# optimizer = optim.SGD(Model.parameters(),lr=float(lr_str))
# rel_optimizer     = optim.Adam(relation_model.parameters(),lr=float(lr_str))
if scheduler_switch:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

start = time.time()
rel_ann_dict = defaultdict(list)


NER_max_scores = [-1 for _ in range(span_size)]


best_average_score = -1
best_re_score = -1

relation_flag = 0
Relation_gold_learning_switch = 1
down_sampling_switch = 0
is_share_stop = False

switches = {}
switches['relation_flag'] = relation_flag
switches['NER_RE_switch'] = NER_RE_switch
switches['down_sampling_switch'] = down_sampling_switch
switches['Relation_gold_learning_switch'] = Relation_gold_learning_switch
switches['TARGET_only_LARGE_NEST_flag'] = TARGET_only_LARGE_NEST_flag
switches['scheduler'] = scheduler_switch
switches['model_save'] = model_save
switches['is_writer'] = is_writer
switches['is_share_stop'] = is_share_stop

def iterate(epoch, data_loader, Model, optimizer, switches, writer):
    Model.eval()
    relation_flag = switches['relation_flag']
    NER_RE_switch = switches['NER_RE_switch']
    down_sampling_switch = switches['down_sampling_switch']
    Relation_gold_learning_switch = switches['Relation_gold_learning_switch']
    scheduler_switch = switches['scheduler']
    is_writer = switches['is_writer']
    is_share_stop = switches['is_share_stop']

    desc = 'test_'

    span_average_loss = 0.0
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
            logits_spans = Model(n_doc, words, attention_mask, relation_flag, NER_RE_switch, down_sampling_switch, y_spans, Relation_gold_learning_switch, is_share_stop)
            for s_x in range(span_size):
                loss_span = loss_functions[s_x](logits_spans[s_x], y_spans.permute(1,0,2)[s_x])
                sum_losses[s_x] += loss_span
                all_loss += loss_span
                span_average_loss += loss_span
                predicts_spans[s_x].append(torch.max(logits_spans[s_x], dim=1)[1])
                answers_spans[s_x].append(y_spans.permute(1,0,2)[s_x])

        if do_re:
            relation_flag = True
            relation_logit, rel_pred_x, rel_y = Model(n_doc, words, attention_mask, relation_flag, NER_RE_switch,down_sampling_switch, y_spans, Relation_gold_learning_switch, is_share_stop)
            loss_relation = loss_function_relation(relation_logit, rel_y)
            sum_loss_relation += loss_relation
            all_loss += loss_relation
            relation_flag = False
            predicts_relation.append(torch.max(relation_logit, 1)[1])
            answers_relation.append(rel_y)


    # pdb.set_trace()


# ここから評価Part
    if is_writer:
        allloss_statement = 'All_loss/' + desc + '/epoch' 
        writer.add_scalar(allloss_statement, all_loss, epoch)
    
    if do_ner:
        print("span average Loss is {0}".format(span_average_loss/span_size))
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
                    print('span {0} Loss is {1}'.format(s_x+1, sum_losses[s_x]))
                    if is_writer:
                        writer.add_scalar(state, sum_losses[s_x], epoch)
                if s > 0 and is_writer:
                    writer.add_scalar(state, micro_p_r_f_spans[s_x][s-1],epoch)
            print(p_r_f_statement + str(micro_p_r_f_spans[s_x]),end='\n\n')

    if do_re:
        golds_relation = torch.cat(answers_relation, 0).cpu().numpy()
        preds_relation = torch.cat(predicts_relation, 0).cpu().numpy()
        micro_p_r_f_relation = precision_recall_fscore_support(golds_relation,preds_relation,labels=[1,2,3,4],average='micro')
        rel_results_per_label = np.array(precision_recall_fscore_support(golds_relation, preds_relation, labels=[1,2,3,4], average=None)).transpose()

        print(desc + 'relation micro p/r/F score is ' + str(micro_p_r_f_relation))
        print(desc + "Loss retlation is {0}".format(sum_loss_relation))
        print(desc + 'positive:',preds_relation.sum(),end='\n\n')
        for l, label in enumerate(rel_results_per_label):
            print('{0} precision, recall, f1value, nums = {1}'.format(get_key_from_value(rel_dic, l+1).ljust(10), label))
        loss_statement = desc + 'relation/loss/epoch'
        precision_statement = desc + 'relation/micro_precision/epoch'
        recall_statement = desc + 'relation/micro_recall/epoch'
        f1_statement = desc + 'relation/micro_f-measure/epoch'
        num_statement = desc + 'relation/number_0f_labels/epoch'
        statement_list = [loss_statement, precision_statement, recall_statement, f1_statement]
        relation_statement_list = [precision_statement, recall_statement, f1_statement, num_statement]

        if is_writer:
            for s, state in enumerate(statement_list):
                if s == 0: 
                    writer.add_scalar(state, loss_relation, epoch)
                if s > 0:
                    writer.add_scalar(state, micro_p_r_f_relation[s-1],epoch)
            # for l, label in enumerate(rel_results_per_label):
            #     rel_log = writer_log_dir+get_key_from_value(rel_dic, l+1) 
            #     rel_writer = tb.SummaryWriter(logdir=rel_log)
            #     for s, state in enumerate(relation_statement_list):
            #         rel_writer.add_scalar(state, label[s], epoch)



epoch = 100
print('\n\n')
joint_switch_controll(epoch,switches)
assert switches['NER_RE_switch'] in ["NER", "RE", "Joint"]
do_re = switches['NER_RE_switch'] in ["RE", "Joint"]
do_ner = switches['NER_RE_switch'] in ["NER", "Joint"]

switches['down_sampling_switch'] = 1
iterate(epoch, test_loader, Model, optimizer, switches, writer)

print_setting()




