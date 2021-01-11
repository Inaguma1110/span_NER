import sys
import os
import argparse
import datetime
import pdb
from pathlib import Path
import itertools
import tqdm
import contextlib
import time
import random
import shelve,pickle
import configparser
from progressbar import progressbar
from collections import defaultdict
from distutils.util import strtobool
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as D
from torch.utils.tensorboard import SummaryWriter

import tensorboardX as tb

from model.BRAN_thesis import JapaneseBertPretrainedModel, MyModel, PairsModule
from util import pred_rel2ann, pred2text, nest_entity_process, get_gold_entity_pair, eval_record_log, NameSpace, relation_weight


np.random.seed(1)
torch.manual_seed(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def p_set(options):
    for k,v in options.__dict__.items():
        print('{0} : {1}'.format(k,v))

def print_mode():
    if switches['NER_RE_switch'] == "NER":
        print("NER Mode")
    if switches['NER_RE_switch'] == "RE":
        print("RE Mode")
        if switches['Relation_gold_learning_switch']:
            print("Relation learned by Gold Entity Data")
        elif not switches['Relation_gold_learning_switch']:
            print("Relation learned by Extracted from NER data")
    if switches['NER_RE_switch'] == "Joint":
        print("NER_RE Joint Mode")
        if switches['Relation_gold_learning_switch']:
            print("Relation learned by Gold Entity Data")
        elif not switches['Relation_gold_learning_switch']:
            print("Relation learned by Extracted from NER data")

    if switches['NER_RE_switch'] == "Joint_ner_share_stop":
        print("NER_RE Joint Mode not optimized share part from NER part")
        if switches['Relation_gold_learning_switch']:
            print("Relation learned by Gold Entity Data")
        elif not switches['Relation_gold_learning_switch']:
            print("Relation learned by Extracted from NER data")

# pdb.set_trace()
config = configparser.ConfigParser()
config.read('../machine_BRAN.conf')
max_sent_len = int(config.get('makedata', 'MAX_SENT_LEN'))

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--experiment', choices=['RE', 'NER', 'Joint_by_preds','sharestop', 'Joint_by_gold','debug','RE_Joint'])
parser.add_argument('--structure',default = "NEST_INNER", help='NEST_INNER -> All nest entity\nNEST_OUTER -> only out nest entity')
parser.add_argument('--target_type',default = "Relation_trigger", help='All -> all sentence   \nRelation   -> only relation sentence')
parser.add_argument('--span_size', default = 4, help = 'Choose span size (default : 4)')
parser.add_argument('--lr', default=1e-4, help='Set learning rate (default : 1e-4)')
parser.add_argument('--train_batch_size', default = 32, help = 'Set train batch size (default : 32)')
parser.add_argument('--batch_shuffle', action='store_true')
parser.add_argument('--is_scheduler',  action='store_true', help = 'Set optimize scheduler (default : False)')
parser.add_argument('--target_only_large_nest_flag', action='store_true' ,help = 'Eval only NEST OUTER (default : False)')
parser.add_argument('--model_save', action='store_true')
parser.add_argument('--is_writer', action='store_true', help = 'If want to record score log, add --is_writer')
parser.add_argument('--relation_init_w', default=1.0)
parser.add_argument('--relation_w_function', choices=['linear','exp','half_linear','zeroone'],default='linear')
parser.add_argument('--spanheadfctype', choices=['each', 'share'],default='share')
parser.add_argument('--embedding_dim', type=int, default=768)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--window_size', type=int, default=3)
parser.add_argument('--h_2head', choices=['SimpleAttention','TrigCatQuery','TrigKeyVal','TrigKeyValResidual'],default='SimpleAttention')
parser.add_argument('--num_MHA',type=int, default=4)
parser.add_argument('--n_epoch', type=int, default=350)
parser.add_argument('--init_Joint_switch', default = 'Joint', help = 'Set mode of init training mode (default : Joint)')
parser.add_argument('--change_epoch')

experiment = parser.parse_args().experiment
data_structure_type = parser.parse_args().structure
target_data_type = parser.parse_args().target_type
span_size = int(parser.parse_args().span_size)
lr = parser.parse_args().lr
batch_size = parser.parse_args().train_batch_size
batch_shuffle = parser.parse_args().batch_shuffle
scheduler_switch = parser.parse_args().is_scheduler
TARGET_only_LARGE_NEST_flag = parser.parse_args().target_only_large_nest_flag
model_save = parser.parse_args().model_save
is_writer = parser.parse_args().is_writer
relation_init_w = float(parser.parse_args().relation_init_w)
relation_w_function = parser.parse_args().relation_w_function
spanheadfctype = parser.parse_args().spanheadfctype
embedding_dim = parser.parse_args().embedding_dim
hidden_dim = parser.parse_args().hidden_dim
window_size = parser.parse_args().window_size
h_2head = parser.parse_args().h_2head
num_MHA = parser.parse_args().num_MHA
n_epoch = parser.parse_args().n_epoch
NER_RE_switch = parser.parse_args().init_Joint_switch
change_epoch = int(parser.parse_args().change_epoch)
model_hparams_dic = {
    'embedding_dim':embedding_dim,
    'hidden_dim':hidden_dim,
    'window_size':window_size,
    'spanheadfctype':spanheadfctype,
    'h_2head':h_2head,
    'num_MHA':num_MHA
    }
args = parser.parse_args()
options = NameSpace.Namespace(**vars(args))
p_set(options)
dt_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d_%H:%M:%S')

brat_log_dir = '../../brat/brat-v1.3_Crunchy_Frog/data/model_preds/span_NER_RE'+dt_now + \
    'Jointmodel_relation/{0}/batch_size_{1}/learning_rate_{2}'.format(
        NER_RE_switch, batch_size, lr)

output_path = Path('../outputs/Last')/ experiment /('change_epoch_is_'+str(change_epoch)) / dt_now


if is_writer:
    writer_log_dir_path = output_path 
    writer = tb.SummaryWriter(logdir=writer_log_dir_path)
else:
    writer = 0

print('change epoch is {0}'.format(change_epoch))

switches = {}
switches['TARGET_only_LARGE_NEST_flag'] = TARGET_only_LARGE_NEST_flag
switches['scheduler'] = scheduler_switch
switches['model_save'] = model_save
switches['is_writer'] = is_writer
switches['is_share_stop'] = False
switches['lr'] = lr
switches['NER_RE_switch'] = NER_RE_switch # NERかREかJointかのモード
switches['Relation_gold_learning_switch'] = 1 # REのための用語ペアをGoldにするかNERのの出力にするかのflag

def joint_switch_controll(epoch, switches,change_epoch=change_epoch):
    global desc_tr, desc_dev, Relation_gold_learning_switch
    if epoch == change_epoch: # Joint
        # if epoch == int(config.get('main', 'START_RELATION_EPOCH')):
        print(str("#" * 80) + '  Start Joint Learning  ' + str("#" * 90), end='\n')
        switches['NER_RE_switch'] = 'Joint'
        switches['is_share_stop'] = False
        desc_tr = "Joint Learning train"
        desc_dev = "Joint Learning devel"
        switches['Relation_gold_learning_switch'] = 0
        # switches['lr'] = switches['lr']* 1e-2
    # if epoch > change_epoch+50:
    #     desc_tr = "Joint Learning train"
    #     desc_dev = "Joint Learning devel"
    #     switches['NER_RE_switch'] = 'Joint'
    #     switches['Relation_gold_learning_switch'] = 1
    #     switches['is_share_stop'] = False

    # if epoch == 40: # Pipeline
    #     # if epoch == int(config.get('main', 'START_RELATION_EPOCH')):
    #     print(str("#" * 80) + '  Start Joint Learning  ' + str("#" * 90), end='\n')
    #     switches['NER_RE_switch'] = "RE"
    #     desc_tr = "Pipe Learning train"
    #     desc_dev = "Pipe Learning devel"
    #     switches['Relation_gold_learning_switch'] = 1
        # switches['is_share_stop'] = True

def repeat_switch(epoch, switches,change_epoch=change_epoch):
    global desc_tr, desc_dev, Relation_gold_learning_switch
    if epoch > change_epoch:
        target = (epoch-change_epoch)//10
        repeater = target % 2

        if repeater == 0:
            print(str("#" * 80) + '  ner_share_stop  ' + str("#" * 90), end='\n')
            switches['NER_RE_switch'] = 'Joint_ner_share_stop'
            switches['is_share_stop'] = False
            desc_tr = "Joint Learning train "
            desc_dev = "Joint Learning devel"
            switches['Relation_gold_learning_switch'] = 1

        if repeater == 1:
            desc_tr = "Joint Learning train"
            desc_dev = "Joint Learning devel"
            switches['NER_RE_switch'] = 'Joint'
            switches['Relation_gold_learning_switch'] = 1
            switches['is_share_stop'] = False

    # if epoch == 40: # Pipeline
    #     # if epoch == int(config.get('main', 'START_RELATION_EPOCH')):
    #     print(str("#" * 80) + '  Start Joint Learning  ' + str("#" * 90), end='\n')
    #     switches['NER_RE_switch'] = "RE"
    #     desc_tr = "Pipe Learning train"
    #     desc_dev = "Pipe Learning devel"
    #     switches['Relation_gold_learning_switch'] = 1
        # switches['is_share_stop'] = True


    print_mode()





def iterate(epoch, data_loader, Model, optimizer, switches, writer, scores, params, best_attention_dict, relation_init_w, is_training):
    NER_best_scores, best_average_score, best_re_score, best_pipeline_re_score = scores
    NER_best_params, RE_best_params, pipeline_RE_best_params = params
    NER_RE_switch = switches['NER_RE_switch']
    Relation_gold_learning_switch = switches['Relation_gold_learning_switch']
    scheduler_switch = switches['scheduler']
    is_writer = switches['is_writer']
    is_share_stop = switches['is_share_stop']
    lr = switches['lr']
    # for g in optimizer.param_groups:
    #     g['lr'] = lr
    if is_training:
        Model.train()
        desc = 'train_'
    else:
        Model.eval()
        desc = 'devel_'
    print('now lr is {}'.format(lr))
    span_average_loss = 0.0
    sum_losses = [0.0 for _ in range(span_size)]
    sum_loss_relation = 0.0
    predicts_spans = [[] for _ in range(span_size)]
    answers_spans  = [[] for _ in range(span_size)]

    predicts_relation = []
    answers_relation = []
    number_gold_entity_pair = [0 for _ in range(len(rel_dic))]

    if do_re:
        relation_weights = torch.ones(len(rel_dic)).to(device)
        relation_weights[0] = relation_weight.Relation_Weight(relation_init_w, epoch-change_epoch).exe(relation_w_function)
        # relation_weights[0] = relation_weight.Relation_Weight(relation_init_w, epoch).exe(relation_w_function)
        if is_writer:
            writer.add_scalar('relation_weight/epoch', relation_weights[0],epoch)

    for i, [n_doc, words, trigger_vecs, padding_mask, y_spans] in enumerate(tqdm.tqdm(data_loader)):
        Model.zero_grad()
        batch_size = words.shape[0]
        all_loss = 0

        
        if switches['NER_RE_switch'] == 'NER':
            logits_spans = Model(n_doc, words, trigger_vecs, padding_mask, NER_RE_switch, y_spans, Relation_gold_learning_switch, is_share_stop=False)
            for s_x in range(span_size):
                loss_span = loss_functions[s_x](logits_spans[s_x], y_spans.permute(1,0,2)[s_x])
                sum_losses[s_x] += loss_span
                all_loss += loss_span
                span_average_loss += loss_span
                predicts_spans[s_x].append(torch.max(logits_spans[s_x], dim=1)[1])
                answers_spans[s_x].append(y_spans.permute(1,0,2)[s_x])

        if switches['NER_RE_switch'] == 'RE':
            relation_logit, trig_attn, rel_y = Model(n_doc, words, trigger_vecs, padding_mask, NER_RE_switch, y_spans, Relation_gold_learning_switch, is_share_stop)
            number_gold_entity_pair = [x+y for (x,y) in zip(number_gold_entity_pair, get_gold_entity_pair.make_label(y_spans, n_doc, REL_LABEL_DICT,rel_dic))]
            loss_relation = F.cross_entropy(relation_logit, rel_y, weight=relation_weights)
            sum_loss_relation += loss_relation
            all_loss += loss_relation
            predicts_relation.append(torch.max(relation_logit, 1)[1])
            answers_relation.append(rel_y)
            attention_dict[str(i)] = [n_doc.to('cpu').detach().numpy().copy(), words.to('cpu').detach().numpy().copy(), trig_attn.to('cpu').detach().numpy().copy(), padding_mask.to('cpu').detach().numpy().copy()]
            

        if switches['NER_RE_switch'] == 'Joint' or switches['NER_RE_switch'] == 'Joint_ner_share_stop':
            logits_spans, relation_logit, trig_attn, rel_y = Model(n_doc, words, trigger_vecs, padding_mask, NER_RE_switch, y_spans, Relation_gold_learning_switch, is_share_stop)
            for s_x in range(span_size):
                loss_span = loss_functions[s_x](logits_spans[s_x], y_spans.permute(1,0,2)[s_x])
                sum_losses[s_x] += loss_span
                all_loss += loss_span
                span_average_loss += loss_span
                predicts_spans[s_x].append(torch.max(logits_spans[s_x], dim=1)[1])
                answers_spans[s_x].append(y_spans.permute(1,0,2)[s_x])

            # number_gold_entity_pair = [x+y for (x,y) in zip(number_gold_entity_pair, get_gold_entity_pair.make_label(nest_entity_process.nest_cut(y_spans,span_size), n_doc, REL_LABEL_DICT,rel_dic))]
            number_gold_entity_pair = [x+y for (x,y) in zip(number_gold_entity_pair, get_gold_entity_pair.make_label(y_spans, n_doc, REL_LABEL_DICT,rel_dic))]
            loss_relation = F.cross_entropy(relation_logit, rel_y, weight=relation_weights)
            sum_loss_relation += loss_relation
            all_loss += (span_size * loss_relation)
            predicts_relation.append(torch.max(relation_logit, 1)[1])
            answers_relation.append(rel_y)
            attention_dict[str(i)] = [n_doc.to('cpu').detach().numpy().copy(), words.to('cpu').detach().numpy().copy(), trig_attn.to('cpu').detach().numpy().copy(), padding_mask.to('cpu').detach().numpy().copy()]

        if is_training:
            all_loss.backward(retain_graph=True)
            optimizer.step()
        if scheduler_switch:
            scheduler.step(all_loss)
    # pdb.set_trace()

    print("span average Loss is {0}".format(span_average_loss/span_size))

    if is_writer:
        allloss_statement = 'All_loss/' + desc + '/epoch' 
        writer.add_scalar(allloss_statement, all_loss, epoch)
    

# ここから評価Part
    if do_ner:
        NER_best_scores, best_average_score, NER_best_params = eval_record_log.eval_spans(span_size, epoch, sum_losses, predicts_spans, answers_spans, NER_best_scores, best_average_score, Model.state_dict(), is_writer, model_save, writer, desc)

    if do_re:
        print('gold ' + '{0}'.format(number_gold_entity_pair))
        print('now realtion None weight {}'.format(relation_weights[0]))
        best_re_score, RE_best_params, best_attention_dict = eval_record_log.eval_relation(rel_dic, epoch, sum_loss_relation, predicts_relation, answers_relation, best_re_score, Model.state_dict(), attention_dict, best_attention_dict, is_writer, model_save, writer, desc)
        best_pipeline_re_score, pipeline_RE_best_params, best_attention_dict = eval_record_log.eval_pipeline_relation(rel_dic, number_gold_entity_pair, epoch, sum_loss_relation, predicts_relation, answers_relation, best_pipeline_re_score, Model.state_dict(), attention_dict, best_attention_dict, is_writer, model_save, writer, desc)
        

    return [NER_best_scores, best_average_score, best_re_score, best_pipeline_re_score], [NER_best_params, RE_best_params, pipeline_RE_best_params], best_attention_dict

############################################################################################################################################
if model_save:
    print('\nModel save Mode\n')
if not model_save:
    print('\nTrain Mode\n')

if target_data_type == "All":
    dataname = str(span_size) + config.get('dataname', 'N_REL_DIVIDED_TRAIN_DEVEL')
if target_data_type == "Relation":
    dataname = str(span_size) + config.get('dataname', 'N_REL_DIVIDED_TRAIN_DEVEL_SHORT')
if target_data_type == 'Relation_trigger':
    dataname = str(span_size) + config.get('dataname', 'N_REL_DIVIDED_TRAIN_DEVEL_TRIGGER')


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
padding_mask = torch.LongTensor([_[3] for _ in corpus]).to(device)
tokenized = [_[4] for _ in corpus]
trigger_vecs = torch.LongTensor([_[5] for _ in corpus]).to(device)


doc_correspnd_info_dict = {}  # document毎にシンボリックな値をdocument名と辞書に変えるための辞書
n_doc = []
for unit in n_Entdics_Reldics:
    doc_correspnd_info_dict[unit[0]] = unit[1:]
    n_doc.append([unit[0]])
n_doc = torch.LongTensor(n_doc).to(device)
print(len(word_input))
dataset = D.TensorDataset(n_doc, word_input, trigger_vecs, padding_mask, output_films)
train_size = int(0.9 * len(word_input))
devel_size = len(word_input) - train_size
print(train_size, devel_size)
train_dataset, devel_dataset = D.random_split(dataset,
                                              [train_size, devel_size])

train_loader = D.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=batch_shuffle)

devel_loader = D.DataLoader(
    devel_dataset,
    batch_size=batch_size,
    shuffle=batch_shuffle)
print('finish', end='\n')


print('Create Model...')
Model = MyModel(span_size, config, vocab, model_hparams_dic, rel_dic, REL_LABEL_DICT, doc_correspnd_info_dict).to(device)
tokenizer = Model.bert_model.tokenizer
print('finish', end='\n\n')

# Loss関数の定義
loss_functions      =nn.ModuleList([nn.CrossEntropyLoss() for _ in range(span_size)])



# Optimizerの設定
optimizer = optim.Adam(Model.parameters(), lr=float(lr),weight_decay=1e-3)
# optimizer = optim.SGD(Model.parameters(),lr=float(lr))
# rel_optimizer     = optim.Adam(relation_model.parameters(),lr=float(lr))
if scheduler_switch:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


#TODO
# attention_dict = shelve.open('../attention_viewing/attention_dict')
# attention_np = np.empty(1)
attention_dict = dict()

rel_ann_dict = defaultdict(list)
tr_NER_best_scores = [[-1,0] for _ in range(span_size)]
tr_best_average_score = [-1,0]
tr_best_re_score = [-1,0]
tr_best_pipeline_re_score = [-1,0]
tr_scores = [tr_NER_best_scores, tr_best_average_score, tr_best_re_score, tr_best_pipeline_re_score]
tr_NER_best_params = 0
tr_RE_best_params=0
tr_pipeline_RE_best_params=0
tr_params=[tr_NER_best_params, tr_RE_best_params, tr_pipeline_RE_best_params]
tr_best_attention_dict ={}


dev_NER_best_scores = [[-1,0] for _ in range(span_size)]
dev_best_average_score  = [-1,0]
dev_best_re_score = [-1,0]
dev_best_pipeline_re_score = [-1,0]
dev_scores = [dev_NER_best_scores, dev_best_average_score, dev_best_re_score, dev_best_pipeline_re_score]
dev_NER_best_params = 0
dev_RE_best_params=0
dev_pipeline_RE_best_params=0
dev_params=[dev_NER_best_params, dev_RE_best_params, dev_pipeline_RE_best_params]
dev_best_attention_dict = {}





for epoch in range(n_epoch):
    print('\n\n')
    print('Current Epoch:{}'.format(epoch + 1))
    joint_switch_controll(epoch,switches)
    # repeat_switch(epoch,switches)
    assert switches['NER_RE_switch'] in ["NER", "RE", "Joint","Joint_ner_share_stop"]
    do_ner = switches['NER_RE_switch'] in ['NER', 'Joint','Joint_ner_share_stop']
    do_re = switches['NER_RE_switch'] in ['RE', 'Joint','Joint_ner_share_stop']
    

    tr_scores, tr_params, tr_best_attention_dict = iterate(epoch, train_loader, Model, optimizer, switches, writer, tr_scores, tr_params, tr_best_attention_dict, relation_init_w, is_training=True)
    
    switches['Relation_gold_learning_switch'] = 1
    dev_scores, dev_params, dev_best_attention_dict = iterate(epoch, devel_loader, Model, optimizer, switches, writer, dev_scores, dev_params, dev_best_attention_dict, relation_init_w, is_training=False)


result=['train : ' +str(tr_scores), 'devel : '+str(dev_scores)]


if not (output_path).exists():
    os.makedirs(output_path)
if not (output_path/'models').exists():
    os.makedirs(output_path/'models')
if not (output_path/'attention_viewing').exists():
    os.makedirs(output_path/'attention_viewing')
(output_path/'result.txt').write_text(str(result))
(output_path/'config.json').write_text(options.json())
with open(output_path/'attention_viewing/attention.binaryfile', 'wb') as aw:
    pickle.dump(dev_best_attention_dict, aw)
if model_save:
    print('save best NER model ... ')
    torch.save(dev_params[0], (output_path/'models'/'NERmodel.pt'))
    
    print('save best RE model ... ')
    torch.save(dev_params[1], (output_path/'models'/'REmodel.pt'))

    print('save best pipeline RE model ... ')
    torch.save(dev_params[1], (output_path/'models'/'pipelineREmodel.pt'))

    print('finish')
p_set(options)