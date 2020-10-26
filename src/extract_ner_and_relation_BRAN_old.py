import sys
import argparse
import datetime
import pdb
import tqdm, contextlib
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

from model.BRAN_old import BERT_PRETRAINED_MODEL_JAPANESE, SPAN_CNN, PAIRS_MODULE, RELATION, BRAN
from util import pred_rel2ann,pred2text,nest_entity_process

def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None

# pdb.set_trace()
parser = argparse.ArgumentParser()
parser.add_argument('--model_save',action='store_true')
model_save = parser.parse_args().model_save
if model_save:
    print('\nModel save Mode\n')
if not model_save:
    print('\nTrain Mode\n')
print('\nCreate Environment...\n')

## reading config file #############################################################
config = configparser.ConfigParser()
config.read('../machine_BRAN.conf')
dt_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d_%H:%M:%S')
print(dt_now)
# dataname          = config.get('dataname', 'REL')  ### SPAN or SPAN_LARGE_NEST
dataname = config.get('dataname', "REL_DIVIDED_TRAIN_DEVEL")
dataname = config.get('dataname', "REL_DIVIDED_TRAIN_DEVEL_SHORT")
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
# writer_log_dir ='../../data/TensorboardGraph/span_NER_RE/Jointmodel/batch_size_{0}/learning_rate_{1}/network_{2}_{3}/0_logit_weight_{4}'.format(batch,lr_str,network_structure,attention_mask_is,weight)
# brat_log_dir   = '../../brat/brat-v1.3_Crunchy_Frog/data/model_preds/span_NER_RE/Jointmodel/batch_size_{0}/learning_rate_{1}/network_{2}_{3}/0_logit_weight_{4}'.format(batch,lr_str,network_structure,attention_mask_is,weight)

writer_log_dir ='../../data/TensorboardGraph/span_Joint/'+dt_now+'Jointmodel_relation/batch_size_{0}/learning_rate_{1}/network_{2}_{3}/0_logit_weight_{4}'.format(batch,lr_str,network_structure,attention_mask_is,weight)
brat_log_dir   = '../../brat/brat-v1.3_Crunchy_Frog/data/model_preds/span_NER_RE'+dt_now+'/Jointmodel_relation/batch_size_{0}/learning_rate_{1}/network_{2}_{3}/0_logit_weight_{4}'.format(batch,lr_str,network_structure,attention_mask_is,weight)



model_writer_log_dir ='../../data/model/'
writer = tb.SummaryWriter(logdir = writer_log_dir)
# model_writer = SummaryWriter()
####################################################################################

np.random.seed(1)
torch.manual_seed(1)
# pdb.set_trace()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


print('\nCreate data...')
database = shelve.open(config.get('path', 'SHELVE_PATH'))
vocab, REL_DIC, corpus, filename_lst,REL_LABEL_DICT = database[dataname] 
database.close()

REL_database = shelve.open(config.get('path', 'REL_DIC_PATH'))
REL_DIC = REL_database['REL_DIC']
REL_database.close()
# pdb.set_trace()
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

# pdb.set_trace()

dataset = D.TensorDataset(n_doc, word_input, attention_mask, y_span_size_1, y_span_size_2, y_span_size_3, y_span_size_4)
train_size = int(0.9 * len(word_input))
devel_size = len(word_input) - train_size 
train_dataset, devel_dataset = D.random_split(dataset, [train_size, devel_size])

train_loader = D.DataLoader(train_dataset, batch_size=int(config.get('main', 'BATCH_SIZE_TRAIN')), shuffle = strtobool(config.get('main', 'BATCH_SHUFFLE_TRAIN')))
devel_loader = D.DataLoader(devel_dataset, batch_size=int(config.get('main', 'BATCH_SIZE_DEVEL')), shuffle = strtobool(config.get('main', 'BATCH_SHUFFLE_DEVEL')))

print('finish', end='\n')
# pdb.set_trace()
print('Create Model...')
tokenizer = BERT_PRETRAINED_MODEL_JAPANESE(config, vocab).return_tokenizer()

model  = SPAN_CNN(config, vocab, REL_DIC).to(device)
# relation_model = BRAN(config, vocab, REL_DIC).to(device)
relation_model = RELATION(config, vocab, REL_DIC).to(device)

print('finish', end='\n\n')


# Loss関数の定義 spanに対して予測のしやすい重みをハイパーパラメータとして定義できる
weights = [float(s) for s in config.get('CNNs', 'LOSS_WEIGHT').split(',')]
class_weights = torch.FloatTensor(weights).to(device)
loss_function_span1 = nn.CrossEntropyLoss(weight=class_weights)
loss_function_span2 = nn.CrossEntropyLoss(weight=class_weights)
loss_function_span3 = nn.CrossEntropyLoss(weight=class_weights)
loss_function_span4 = nn.CrossEntropyLoss(weight=class_weights)

loss_function_relation = nn.CrossEntropyLoss(ignore_index=0)

optimizer     = optim.Adam(model.parameters(), lr=float(lr_str))
rel_optimizer     = optim.Adam(relation_model.parameters(),lr=float(lr_str))
if strtobool(config.get('main', 'SCHEDULER')):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    rel_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(rel_optimizer, 'min')




print('target data is ' + dataname, end = '\n\n')
print('batch size is {0}'.format(batch))
print('Number of train, develop is {0}, {1}'.format(train_size, devel_size))
print('learning rate is  {0}'.format(lr_str))
print('network strucuture is {0}'.format(network_structure))
print('iteration is {0}'.format(config.get('main', 'N_EPOCH')))
print('Start Relation Epoch is {0}'.format(config.get('main', 'START_RELATION_EPOCH')))
print('0_logit weight is {0}'.format(weight), end = '\n\n')
print('Scheduler is {0}'.format(strtobool(config.get('main', 'SCHEDULER'))))

print('tensorboard writer logdir is {0}'.format(writer_log_dir))
print('brat logdir is {0}'.format(brat_log_dir))
print('Start Training... ',end ='\n\n')

start = time.time()
model.train()
# pdb.set_trace()


span1_max_f = -1
span2_max_f = -1
span3_max_f = -1
span4_max_f = -1
best_average_score = -1
best_re_score = -1


relation_flag = 0


NER_max_scores = [span1_max_f, span2_max_f, span3_max_f, span4_max_f]
for epoch in range(int(config.get('main', 'N_EPOCH'))):
    # pdb.set_trace()
    if epoch == int(config.get('main', 'START_RELATION_EPOCH')):
        relation_flag = True
    print('Current Epoch:{}'.format(epoch+1))
    data_unit_for_relation = []
    dict_unit_for_relation = {0:0,1:0,2:0,3:0,4:0,5:0}
    Num_of_rel_label = 0
    model.train()
    sum_loss_tr = 0.0

    for i, [n_doc, words, attention_mask, y_span_size_1, y_span_size_2, y_span_size_3, y_span_size_4] in enumerate(tqdm.tqdm(train_loader,desc="Entity Recognition train")):
        pred_pair_unit = []
        model.zero_grad()
        # pdb.set_trace()
        logits_span1, logits_span2, logits_span3, logits_span4 = model(words, attention_mask, relation_flag, pred_pair_unit)
        loss_span1       = loss_function_span1(logits_span1, y_span_size_1)
        loss_span2       = loss_function_span2(logits_span2, y_span_size_2)
        loss_span3       = loss_function_span3(logits_span3, y_span_size_3)
        loss_span4       = loss_function_span4(logits_span4, y_span_size_4)
        loss = loss_span1 + loss_span2 + loss_span3 + loss_span4
        loss.backward(retain_graph=True)
        optimizer.step()
        # pdb.set_trace()


        if relation_flag == True: # Relationの学習
            # pdb.set_trace()
            predicts_span1 = []
            predicts_span2 = []
            predicts_span3 = []
            predicts_span4 = []

            predicts_span1.append(torch.max(logits_span1, 1)[1])
            predicts_span2.append(torch.max(logits_span2, 1)[1])
            predicts_span3.append(torch.max(logits_span3, 1)[1])
            predicts_span4.append(torch.max(logits_span4, 1)[1])

            # goldで実験するとき
            # predicts_span1.append(y_span_size_1)
            # predicts_span2.append(y_span_size_2)
            # predicts_span3.append(y_span_size_3)
            # predicts_span4.append(y_span_size_4)


            # predicts_span1,predicts_span2,predicts_span3,predicts_span4 = nest_entity_process.nest_square_cut(predicts_span1,predicts_span2,predicts_span3,predicts_span4)
            pred_pairs_list_per_batch = PAIRS_MODULE(predicts_span1,predicts_span2,predicts_span3,predicts_span4).MAKE_PAIR()
            # REのDatasetを作る = 予測したペアに対してラベルを振る
            for number_in_minibatch, unique_number in enumerate(n_doc):
                pred_pairs = pred_pairs_list_per_batch[number_in_minibatch]
                gold_pairs = REL_LABEL_DICT[unique_number.item()]
                Num_of_rel_label += len(gold_pairs)
                for pred_pair_unit in pred_pairs:
                    rel_label_index = REL_DIC["None"]
                    for gold_pair_unit in gold_pairs:
                        shaped_gold_unit = [(gold_pair_unit[1][0][0],gold_pair_unit[1][0][1][0]),(gold_pair_unit[1][1][0],gold_pair_unit[1][1][1][0])]
                        if set(pred_pair_unit) == set(shaped_gold_unit):
                            # rel_label_index = gold_pair_unit[0]
                            rel_label_index = 1
                            dict_unit_for_relation[rel_label_index] += 1
                    relation_entity_all_label = (unique_number, words[number_in_minibatch].cpu().numpy(), attention_mask[number_in_minibatch].cpu().numpy(), pred_pair_unit, rel_label_index)
                    data_unit_for_relation.append(relation_entity_all_label)
    if relation_flag == True:
        print("make train relation data ... ")
        unique_x             = torch.LongTensor([a[0] for a in data_unit_for_relation]).to(device)
        rel_word_x           = torch.LongTensor([a[1] for a in data_unit_for_relation]).to(device)
        rel_attention_mask   = torch.LongTensor([a[2] for a in data_unit_for_relation]).to(device)
        rel_pred_x           = torch.LongTensor([a[3] for a in data_unit_for_relation]).to(device)
        rel_y                = torch.LongTensor([a[4] for a in data_unit_for_relation]).to(device)
        
        nonzero_indexes = rel_y.nonzero().squeeze().cpu().numpy().tolist()
        zero_indexes = []
        for index in range(0,len(rel_y)):
            if index in nonzero_indexes:
                pass
            else:
                zero_indexes.append(index)
        # pdb.set_trace()
        zero_indexes = random.sample(zero_indexes, int(float(len(nonzero_indexes)*0.25)))

        unique_x_list = []
        rel_word_x_list = []
        rel_attention_mask_list = []
        rel_pred_x_list = []
        rel_y_list = []
        for c in range(0,len(data_unit_for_relation)):
            if c in nonzero_indexes or c in zero_indexes:
                unique_x_list.append(data_unit_for_relation[c][0])
                rel_word_x_list.append(data_unit_for_relation[c][1])
                rel_attention_mask_list.append(data_unit_for_relation[c][2])
                rel_pred_x_list.append(data_unit_for_relation[c][3])
                rel_y_list.append(data_unit_for_relation[c][4])
        unique_x = torch.LongTensor(unique_x_list).to(device)
        rel_word_x = torch.LongTensor(rel_word_x_list).to(device)
        rel_attention_mask = torch.LongTensor(rel_attention_mask_list).to(device)
        rel_pred_x = torch.LongTensor(rel_pred_x_list).to(device)
        rel_y = torch.LongTensor(rel_y_list).to(device)


        reldataset = D.TensorDataset(unique_x, rel_word_x, rel_attention_mask, rel_pred_x, rel_y)
        rel_train_loader = D.DataLoader(reldataset, batch_size = int(config.get('main', 'REL_BATCH_SIZE_TRAIN')), shuffle = strtobool(config.get('main', 'BATCH_SHUFFLE_TRAIN')))
        print("finish")
        print("Number of relation data is {0}".format(len(rel_y)))
        print("Number of Nonzero label is {0}".format(len(rel_y.nonzero())))
        print("Number of train gold Nonzero relation label is {0}".format(Num_of_rel_label))
        rel_res_list=[]
        rel_rate = 0
        for rel_key,rel_index in REL_DIC.items():
            rel_res_list.append((rel_key, dict_unit_for_relation[rel_index]))
            rel_rate += dict_unit_for_relation[rel_index]
        print("Label Train breakdown {0}".format(rel_res_list))
        print("Laebl Train recall {0}/{1} = {2}".format(rel_rate,Num_of_rel_label,(rel_rate/Num_of_rel_label)*100))

        relation_model.train()
        for r, [unique_x, rel_word_x, rel_attention_mask, rel_pred_x, rel_y] in enumerate(tqdm.tqdm(rel_train_loader, desc="Relation Extraction train")):
            relation_model.zero_grad()
            relation_logit = relation_model(rel_word_x, rel_attention_mask, rel_pred_x, 0)
            loss_relation = loss_function_relation(relation_logit, rel_y)
            loss_relation.backward(retain_graph=False)
            rel_optimizer.step()

        # pdb.set_trace()





    ########################################################## Develop process ###########################################################
    
    print(str("#" * 25) + '  develop process  ' + str("#" * 25),end='\n')
    sum_loss = 0.0
    sum_loss_span1 = 0.0
    sum_loss_span2 = 0.0
    sum_loss_span3 = 0.0
    sum_loss_span4 = 0.0
    predicts_span1 = []
    answers_span1  = []
    predicts_span2 = []
    answers_span2  = []
    predicts_span3 = []
    answers_span3  = []
    predicts_span4 = []
    answers_span4  = []
    predicts_relation  = []
    answers_relation   = []
    data_unit_for_relation = []
    dict_unit_for_relation = {0:0,1:0,2:0,3:0,4:0,5:0}
    # pdb.set_trace()

    Num_of_rel_label = 0
    for i, [n_doc, words, attention_mask, y_span_size_1, y_span_size_2, y_span_size_3, y_span_size_4] in enumerate(tqdm.tqdm(devel_loader,desc="Entity Recognition develop")):
        # pdb.set_trace()
        model.eval()
        batch_size = words.shape[0]

        # pdb.set_trace()
        pred_pair_unit = []
        model.zero_grad()
        logits_span1, logits_span2, logits_span3, logits_span4 = model(words, attention_mask, relation_flag, pred_pair_unit)
        loss_span1       = loss_function_span1(logits_span1, y_span_size_1)
        loss_span2       = loss_function_span2(logits_span2, y_span_size_2)
        loss_span3       = loss_function_span3(logits_span3, y_span_size_3)
        loss_span4       = loss_function_span4(logits_span4, y_span_size_4)
        loss = loss_span1 + loss_span2 + loss_span3 + loss_span4

        if strtobool(config.get('main', 'SCHEDULER')):
            scheduler.step(loss)
        sum_loss_span1  += float(loss_span1) * batch_size
        sum_loss_span2  += float(loss_span2) * batch_size
        sum_loss_span3  += float(loss_span3) * batch_size
        sum_loss_span4  += float(loss_span4) * batch_size

        sum_loss  += float(loss) * batch_size

        predicts_span1.append(torch.max(logits_span1, 1)[1])
        predicts_span2.append(torch.max(logits_span2, 1)[1])
        predicts_span3.append(torch.max(logits_span3, 1)[1])
        predicts_span4.append(torch.max(logits_span4, 1)[1])

        answers_span1.append(y_span_size_1)
        answers_span2.append(y_span_size_2)
        answers_span3.append(y_span_size_3)
        answers_span4.append(y_span_size_4)


        if relation_flag == True: # Relationの学習
            pred_pairs_list_per_batch = PAIRS_MODULE(predicts_span1,predicts_span2,predicts_span3,predicts_span4).MAKE_PAIR()
            # REのDatasetを作る = 予測したペアに対してラベルを振る
            for number_in_minibatch, unique_number in enumerate(n_doc):
                pred_pairs = pred_pairs_list_per_batch[number_in_minibatch]
                gold_pairs = REL_LABEL_DICT[unique_number.item()]
                Num_of_rel_label += len(gold_pairs)
                for pred_pair_unit in pred_pairs:
                    rel_label_index = REL_DIC["None"]
                    for gold_pair_unit in gold_pairs:
                        shaped_gold_unit = [(gold_pair_unit[1][0][0],gold_pair_unit[1][0][1][0]),(gold_pair_unit[1][1][0],gold_pair_unit[1][1][1][0])]
                        if set(pred_pair_unit) == set(shaped_gold_unit):
                            # rel_label_index = gold_pair_unit[0]
                            rel_label_index = 1
                            dict_unit_for_relation[rel_label_index] += 1
                    relation_entity_all_label = (unique_number, words[number_in_minibatch].cpu().numpy(), attention_mask[number_in_minibatch].cpu().numpy(), pred_pair_unit, rel_label_index)
                    data_unit_for_relation.append(relation_entity_all_label)
            # pdb.set_trace()
    
    if relation_flag == True:
        # pdb.set_trace()
        print("make devel relation data ... ")
        unique_x             = torch.LongTensor([a[0] for a in data_unit_for_relation]).to(device)
        rel_word_x           = torch.LongTensor([a[1] for a in data_unit_for_relation]).to(device)
        rel_attention_mask   = torch.LongTensor([a[2] for a in data_unit_for_relation]).to(device)
        rel_pred_x           = torch.LongTensor([a[3] for a in data_unit_for_relation]).to(device)
        rel_y                = torch.LongTensor([a[4] for a in data_unit_for_relation]).to(device)

        # nonzero_indexes = rel_y.nonzero().squeeze().cpu().numpy().tolist()
        # zero_indexes = []
        # for index in range(0,len(rel_y)):
        #     if index in nonzero_indexes:
        #         pass
        #     else:
        #         zero_indexes.append(index)
        # # pdb.set_trace()
        # zero_indexes = random.sample(zero_indexes, int(float(len(nonzero_indexes)*0.5)))

        # unique_x_list = []
        # rel_word_x_list = []
        # rel_attention_mask_list = []
        # rel_pred_x_list = []
        # rel_y_list = []
        # for c in range(0,len(data_unit_for_relation)):
        #     if c in nonzero_indexes or c in zero_indexes:
        #         unique_x_list.append(data_unit_for_relation[c][0])
        #         rel_word_x_list.append(data_unit_for_relation[c][1])
        #         rel_attention_mask_list.append(data_unit_for_relation[c][2])
        #         rel_pred_x_list.append(data_unit_for_relation[c][3])
        #         rel_y_list.append(data_unit_for_relation[c][4])
        # unique_x = torch.LongTensor(unique_x_list).to(device)
        # rel_word_x = torch.LongTensor(rel_word_x_list).to(device)
        # rel_attention_mask = torch.LongTensor(rel_attention_mask_list).to(device)
        # rel_pred_x = torch.LongTensor(rel_pred_x_list).to(device)
        # rel_y = torch.LongTensor(rel_y_list).to(device)


        reldataset = D.TensorDataset(unique_x, rel_word_x, rel_attention_mask, rel_pred_x, rel_y)
        rel_devel_loader = D.DataLoader(reldataset, batch_size = int(config.get('main', 'REL_BATCH_SIZE_TRAIN')), shuffle = strtobool(config.get('main', 'BATCH_SHUFFLE_TRAIN')))
        print("finish")
        print("Number of devel relation dataset is {0} ".format(len(rel_y)))
        print("Number of Nonzero label is {0}".format(len(rel_y.nonzero())))
        print("Number of devel gold Nonzero relation label is {0}".format(Num_of_rel_label))
        rel_res_list=[]
        rel_rate = 0
        for rel_key,rel_index in REL_DIC.items():
            rel_res_list.append((rel_key, dict_unit_for_relation[rel_index]))
            rel_rate += dict_unit_for_relation[rel_index]
        print("Label Devel breakdown {0}".format(rel_res_list))
        print("Laebl Devel recall {0}/{1} = {2}".format(rel_rate,Num_of_rel_label,(rel_rate/Num_of_rel_label)*100))
        writer.add_scalar("relation_dataset/Extracted_number_of_pair/epoch", len(rel_y),epoch)
        writer.add_scalar("relation_dataset/Made_Nonzero_label/epoch", len(rel_y.nonzero()),epoch)
        writer.add_scalar("relation_dataset/Made_Nonzero_recall/epoch", (rel_rate/Num_of_rel_label)*100,epoch)
        writer.add_scalar("relation_dataset/Relation/epoch", dict_unit_for_relation[1],epoch)
        writer.add_scalar("relation_dataset/Negative/epoch", dict_unit_for_relation[2],epoch)
        writer.add_scalar("relation_dataset/Positive/epoch", dict_unit_for_relation[3],epoch)
        writer.add_scalar("relation_dataset/Sub/epoch", dict_unit_for_relation[4],epoch)
        writer.add_scalar("relation_dataset/Part/epoch", dict_unit_for_relation[5],epoch)




        relation_model.eval()
        for r, [unique_x, rel_word_x, rel_attention_mask, rel_pred_x, rel_y] in enumerate(tqdm.tqdm(rel_devel_loader, desc="Relation Extraction develop")):
            debugflag = 0
            # if r == len(rel_devel_loader)-1: 
            #     debugflag = 1
            relation_model.zero_grad()
            relation_logit = relation_model(rel_word_x, rel_attention_mask, rel_pred_x, debugflag)
            loss_relation = loss_function_relation(relation_logit, rel_y)
            if strtobool(config.get('main', 'SCHEDULER')):
                rel_scheduler.step(loss_relation)
            predicts_relation.append(torch.max(relation_logit, 1)[1])
            answers_relation.append(rel_y)
            # pdb.set_trace()


    # pdb.set_trace()
    print('calculate span score ...')
    preds_span1 = torch.cat(predicts_span1, 0).view(-1,1).squeeze().cpu().numpy()
    golds_span1 = torch.cat(answers_span1, 0).view(-1,1).squeeze().cpu().numpy()

    preds_span2 = torch.cat(predicts_span2, 0).view(-1,1).squeeze().cpu().numpy()
    golds_span2 = torch.cat(answers_span2, 0).view(-1,1).squeeze().cpu().numpy()

    preds_span3 = torch.cat(predicts_span3, 0).view(-1,1).squeeze().cpu().numpy()
    golds_span3 = torch.cat(answers_span3, 0).view(-1,1).squeeze().cpu().numpy()

    preds_span4 = torch.cat(predicts_span4, 0).view(-1,1).squeeze().cpu().numpy()
    golds_span4 = torch.cat(answers_span4, 0).view(-1,1).squeeze().cpu().numpy()


    ######## nestの外部を評価するために内部の予測を0にしてから評価するための処理 ######
    if TARGET_only_LARGE_NEST_flag: 
        preds_span1,preds_span2,preds_span3,preds_span4 = nest_entity_process.nest_square_cut_for_eval(preds_span1,preds_span2,preds_span3,preds_span4)
        golds_span1,golds_span2,golds_span3,golds_span4 = nest_entity_process.nest_square_cut_for_eval(golds_span1,golds_span2,golds_span3,golds_span4)

    
    #########################################################################

    micro_p_r_f_span1 = precision_recall_fscore_support(golds_span1,preds_span1,labels=[1], average='micro')    
    micro_p_r_f_span2 = precision_recall_fscore_support(golds_span2,preds_span2,labels=[1], average='micro')    
    micro_p_r_f_span3 = precision_recall_fscore_support(golds_span3,preds_span3,labels=[1], average='micro')
    micro_p_r_f_span4 = precision_recall_fscore_support(golds_span4,preds_span4,labels=[1], average='micro')


    print('span1 micro p/r/F score is ' + str(micro_p_r_f_span1))
    print('span2 micro p/r/F score is ' + str(micro_p_r_f_span2))
    print('span3 micro p/r/F score is ' + str(micro_p_r_f_span3))
    print('span4 micro p/r/F score is ' + str(micro_p_r_f_span4))
    print()


    if span1_max_f < micro_p_r_f_span1[2]:
        span1_max_scores = (epoch,micro_p_r_f_span1)
        span1_max_f = micro_p_r_f_span1[2]
        NER_max_scores[0] = span1_max_scores

    if span2_max_f < micro_p_r_f_span2[2]:
        span2_max_scores = (epoch,micro_p_r_f_span2)
        span2_max_f = micro_p_r_f_span2[2]
        NER_max_scores[1] = span2_max_scores

    if span3_max_f < micro_p_r_f_span3[2]:
        span3_max_scores = (epoch,micro_p_r_f_span3)
        span3_max_f = micro_p_r_f_span3[2]
        NER_max_scores[2] = span3_max_scores

    if span4_max_f < micro_p_r_f_span4[2]:
        span4_max_scores = (epoch,micro_p_r_f_span4)
        span4_max_f = micro_p_r_f_span4[2]
        NER_max_scores[3] = span4_max_scores

    average_score = sum(list([a for a in zip(micro_p_r_f_span1,micro_p_r_f_span2,micro_p_r_f_span3,micro_p_r_f_span4)][2]))/4
    print('span average score is {0} epoch is {1}'.format(average_score, epoch),end='\n\n\n')
    if average_score > best_average_score:
        best_average_score = average_score
        if model_save:
            torch.save(model.state_dict(),NER_model_save_path)
    
    if relation_flag:
        print("calculate relation score ...")
        micro_p_r_f_relation = precision_recall_fscore_support(answers_relation[0].cpu().numpy(),predicts_relation[0].cpu().numpy(), labels=[1], average='micro')
        print('relation micro p/r/F score is ' + str(micro_p_r_f_relation) ,end='\n\n\n\n')


        if best_re_score < micro_p_r_f_relation[2]:
            best_re_score = micro_p_r_f_relation[2]
            if model_save:
                torch.save(relation_model.state_dict(),RE_model_save_path)



    writer.add_scalar('span1_devel/loss_span1/epoch', sum_loss_span1, epoch)
    writer.add_scalar('span1_devel/micro_precision/epoch', micro_p_r_f_span1[0],epoch)
    writer.add_scalar('span1_devel/micro_recall/epoch', micro_p_r_f_span1[1],epoch)
    writer.add_scalar('span1_devel/micro_f-measure/epoch', micro_p_r_f_span1[2],epoch)

    writer.add_scalar('span2_devel/loss_span2/epoch', sum_loss_span2, epoch)
    writer.add_scalar('span2_devel/micro_precision/epoch', micro_p_r_f_span2[0],epoch)
    writer.add_scalar('span2_devel/micro_recall/epoch', micro_p_r_f_span2[1],epoch)
    writer.add_scalar('span2_devel/micro_f-measure/epoch', micro_p_r_f_span2[2],epoch)

    writer.add_scalar('span3_devel/loss_span3/epoch', sum_loss_span3, epoch)
    writer.add_scalar('span3_devel/micro_precision/epoch', micro_p_r_f_span3[0],epoch)
    writer.add_scalar('span3_devel/micro_recall/epoch', micro_p_r_f_span3[1],epoch)
    writer.add_scalar('span3_devel/micro_f-measure/epoch', micro_p_r_f_span3[2],epoch)

    writer.add_scalar('span4_devel/loss_span4/epoch', sum_loss_span4, epoch)
    writer.add_scalar('span4_devel/micro_precision/epoch', micro_p_r_f_span4[0],epoch)
    writer.add_scalar('span4_devel/micro_recall/epoch', micro_p_r_f_span4[1],epoch)
    writer.add_scalar('span4_devel/micro_f-measure/epoch', micro_p_r_f_span4[2],epoch)

    if relation_flag:    

        writer.add_scalar('relation/loss/epoch', sum_loss, epoch)
        writer.add_scalar('relation/micro_precision/epoch', micro_p_r_f_relation[0],epoch)
        writer.add_scalar('relation/micro_recall/epoch', micro_p_r_f_relation[1],epoch)
        writer.add_scalar('relation/micro_f-measure/epoch', micro_p_r_f_relation[2],epoch)











    if (epoch+1) % 1000 == 0:
        print('\n start test...\n')
        model.eval()
        predicted_tokens = []
        n_docs = []

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

        # pdb.set_trace()
        with torch.no_grad():
            for i, [n_doc, words, attention_mask, y_span_size_1, y_span_size_2, y_span_size_3, y_span_size_4] in enumerate(test_loader):
                # pdb.set_trace()
                _, logits_span1, logits_span2, logits_span3, logits_span4 = model(words, attention_mask, relation_flag)

                predicts_span1.append(torch.max(logits_span1, 1)[1])
                predicts_span2.append(torch.max(logits_span2, 1)[1])
                predicts_span3.append(torch.max(logits_span3, 1)[1])
                predicts_span4.append(torch.max(logits_span4, 1)[1])

                answers_span1.append(y_span_size_1)
                answers_span2.append(y_span_size_2)
                answers_span3.append(y_span_size_3)
                answers_span4.append(y_span_size_4)
                
                # pdb.set_trace()






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

                # for num_unit in range(0, words.size(0)):
                #     answers_token_preds_golds1.append((predicted_tokens[num_unit], predicts_span1[0][num_unit], answers_span1[0][num_unit]))
                #     answers_token_preds_golds2.append((predicted_tokens[num_unit], predicts_span2[0][num_unit], answers_span2[0][num_unit]))
                #     answers_token_preds_golds3.append((predicted_tokens[num_unit], predicts_span3[0][num_unit], answers_span3[0][num_unit]))
                #     answers_token_preds_golds4.append((predicted_tokens[num_unit], predicts_span4[0][num_unit], answers_span4[0][num_unit]))

            # pdb.set_trace()
            print('predictions -> annotations\n')
            pred_rel2ann.pred_rel2ann(epoch+1, brat_log_dir, doc_correspnd_info_dict, n_docs, predicted_tokens, for_txt_span1, for_txt_span2, for_txt_span3, for_txt_span4)

            print('{0} Test calculate NER score ... {0}'.format('#'*30))
            print('calculate NER score ...')

            preds_span1 = torch.cat(predicts_span1, 0).view(-1,1).squeeze().cpu().numpy()
            golds_span1 = torch.cat(answers_span1, 0).view(-1,1).squeeze().cpu().numpy()

            preds_span2 = torch.cat(predicts_span2, 0).view(-1,1).squeeze().cpu().numpy()
            golds_span2 = torch.cat(answers_span2, 0).view(-1,1).squeeze().cpu().numpy()

            preds_span3 = torch.cat(predicts_span3, 0).view(-1,1).squeeze().cpu().numpy()
            golds_span3 = torch.cat(answers_span3, 0).view(-1,1).squeeze().cpu().numpy()

            preds_span4 = torch.cat(predicts_span4, 0).view(-1,1).squeeze().cpu().numpy()
            golds_span4 = torch.cat(answers_span4, 0).view(-1,1).squeeze().cpu().numpy()



            ######## nestの外部を評価するために内部の予測を0にしてから評価するための処理 ######
            if TARGET_only_LARGE_NEST_flag: 
                for indx in range(0, len(preds_span4)):
                    try:
                        if preds_span4[indx] == 1:
                            preds_span3[indx] = preds_span3[indx+1] = preds_span3[indx+2] = preds_span3[indx+3] = 0
                            preds_span2[indx] = preds_span2[indx+1] = preds_span2[indx+2] = preds_span2[indx+3] = 0
                            preds_span1[indx] = preds_span1[indx+1] = preds_span1[indx+2] = preds_span1[indx+3] = 0
                            continue
                        else:
                            if preds_span3[indx] == 1:
                                preds_span2[indx] = preds_span2[indx+1] = preds_span2[indx+2] = 0
                                preds_span1[indx] = preds_span1[indx+1] = preds_span1[indx+2] = 0
                                continue
                            else:
                                if preds_span2[indx] == 1:
                                    preds_span1[indx] = preds_span1[indx+1] = 0
                                    continue
                    except IndexError:
                        pass


                for indx in range(0, len(golds_span4)):
                    try:
                        if golds_span4[indx] == 1:
                            golds_span3[indx] = golds_span3[indx+1] = golds_span3[indx+2] = golds_span3[indx+3] = 0
                            golds_span2[indx] = golds_span2[indx+1] = golds_span2[indx+2] = golds_span2[indx+3] = 0
                            golds_span1[indx] = golds_span1[indx+1] = golds_span1[indx+2] = golds_span1[indx+3] = 0
                            continue
                        else:
                            if golds_span3[indx] == 1:
                                golds_span2[indx] = golds_span2[indx+1] = golds_span2[indx+2] = 0
                                golds_span1[indx] = golds_span1[indx+1] = golds_span1[indx+2] = 0
                                continue
                            else:
                                if golds_span2[indx] == 1:
                                    golds_span1[indx] = golds_span1[indx+1] = 0
                                    continue
                    except IndexError:
                        pass
            ####################################################################################

            micro_p_r_f_span1 = precision_recall_fscore_support(golds_span1,preds_span1,labels=[1], average='micro')    
            micro_p_r_f_span2 = precision_recall_fscore_support(golds_span2,preds_span2,labels=[1], average='micro')    
            micro_p_r_f_span3 = precision_recall_fscore_support(golds_span3,preds_span3,labels=[1], average='micro')
            micro_p_r_f_span4 = precision_recall_fscore_support(golds_span4,preds_span4,labels=[1], average='micro')

            print('span1 micro p/r/F score is ' + str(micro_p_r_f_span1))
            print('span2 micro p/r/F score is ' + str(micro_p_r_f_span2))
            print('span3 micro p/r/F score is ' + str(micro_p_r_f_span3))
            print('span4 micro p/r/F score is ' + str(micro_p_r_f_span4))
            print('\n')

        model.train()

print('span max develop score is ' + str(NER_max_scores), end='\n\n')
print('elapsed time is {0}'.format(time.time() - start))


# print('RE max score is ' + str(RE_max_scores))
