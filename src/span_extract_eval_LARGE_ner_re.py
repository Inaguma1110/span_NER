import sys
import argparse
import pdb
import tqdm, contextlib
import time
import shelve
import configparser
from distutils.util import strtobool
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as D

import tensorboardX as tb

from model.CNN import BERT_PRETRAINED_MODEL_JAPANESE, SPAN_CNN
from util import pred2ann


config = configparser.ConfigParser()
config.read('../machine.conf')

dataname          = config.get('dataname', 'SPAN')  ### SPAN or SPAN_LARGE_NEST
max_sent_len      = int(config.get('makedata', 'MAX_SENT_LEN'))
network_structure = config.get('CNNs', 'NETWORK_STRUCTURE')
weight            = config.get('CNNs', 'WEIGHT')
batch             = config.get('main', 'BATCH_SIZE_TRAIN')
lr_str            = config.get('main', 'LEARNING_RATE')
attention_mask_is = config.get('CNNs', 'ATTENTION_MASK_IS')
TARGET_only_LARGE_NEST_flag        = strtobool(config.get('main','NEST'))

writer_log_dir ='../../data/TensorboardGraph/span_NER_RE/correcteval_answerchanged2-4_TARGET_only_LARGE_NEST_flag_is_{5}/batch_size_{0}/learning_rate_{1}/network_{2}_{3}/0_logit_weight_{4}'.format(batch,lr_str,network_structure,attention_mask_is,weight,str(TARGET_only_LARGE_NEST_flag))
brat_log_dir   = '../../brat/brat-v1.3_Crunchy_Frog/data/model_preds/span_NER_RE/correcteval_answerchanged2-4_TARGET_only_LARGE_NEST_flag_is_{5}/batch_size_{0}/learning_rate_{1}/network_{2}_{3}/0_logit_weight_{4}'.format(batch,lr_str,network_structure,attention_mask_is,weight,str(TARGET_only_LARGE_NEST_flag))

hoge_dir = '../../data/TensorboardGraph/span_NER_RE/LARGE_NEST_{}_debug'.format(str(TARGET_only_LARGE_NEST_flag))

writer = tb.SummaryWriter(logdir = writer_log_dir)

np.random.seed(1)
torch.manual_seed(1)
# pdb.set_trace()

print('\nCreate Environment...\n')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('\nCreate data...')
database = shelve.open(config.get('path', 'SHELVE_PATH'))
# pdb.set_trace()
vocab, REL_DIC, corpus, filename_lst = database[dataname] 
database.close()

# (doc[0], indx_tokens, output_film_size1, output_film_size2, output_film_size3, output_film_size4, attention_mask, spmed, (n,doc,Entdic, Reldic))
document            = [a[0] for a in corpus]
word_input          = torch.LongTensor([a[1] for a in corpus]).to(device)
y_span_size_1       = torch.LongTensor([a[2] for a in corpus]).to(device) 
y_span_size_2       = torch.LongTensor([a[3] for a in corpus]).to(device)
y_span_size_3       = torch.LongTensor([a[4] for a in corpus]).to(device)
y_span_size_4       = torch.LongTensor([a[5] for a in corpus]).to(device)
attention_mask      = torch.LongTensor([a[6] for a in corpus]).to(device)
sentencepieced      = [a[7] for a in corpus]
n_Entdics_Reldics   = [a[8] for a in corpus]

doc_correspnd_info_dict = {} #document毎にシンボリックな値をdocument名と辞書に変えるための辞書

n_doc = []
for unit in n_Entdics_Reldics:
    doc_correspnd_info_dict[unit[0]] = unit[1:]
    n_doc.append([unit[0]])
n_doc = torch.LongTensor(n_doc).to(device)

# pdb.set_trace()
dataset = D.TensorDataset(n_doc, word_input, attention_mask, y_span_size_1, y_span_size_2, y_span_size_3, y_span_size_4)
train_size = int(0.8 * len(word_input))
devel_size = int(0.1 * len(word_input))
test_size = len(word_input) - train_size - devel_size
train_dataset, devel_dataset, test_dataset = D.random_split(dataset, [train_size, devel_size, test_size])
train_loader = D.DataLoader(train_dataset, batch_size=int(config.get('main', 'BATCH_SIZE_TRAIN')), shuffle = strtobool(config.get('main', 'BATCH_SHUFFLE_TRAIN')))
devel_loader = D.DataLoader(devel_dataset, batch_size=int(config.get('main', 'BATCH_SIZE_DEVEL')), shuffle = strtobool(config.get('main', 'BATCH_SHUFFLE_DEVEL')))
test_loader  = D.DataLoader(test_dataset , batch_size=int(config.get('main', 'BATCH_SIZE_TEST' )), shuffle = strtobool(config.get('main', 'BATCH_SHUFFLE_TEST')))

print('finish', end='\n')

print('Create Model...')
bert_pretrained_model_japanese = BERT_PRETRAINED_MODEL_JAPANESE(config, vocab)
tokenizer = bert_pretrained_model_japanese.return_tokenizer()
pretrained_BERT_model = bert_pretrained_model_japanese.return_model()
# tokenizer.convert_ids_to_tokens()というメソッドでindexを用語に直せる
model  = SPAN_CNN(config, vocab, REL_DIC).to(device)
print('finish', end='\n\n')

# Loss関数の定義 spanに対して予測のしやす重みをハイパーパラメータとして定義できる
weights = [float(s) for s in config.get('CNNs', 'LOSS_WEIGHT').split(',')]
class_weights = torch.FloatTensor(weights).to(device)
loss_function_span1 = nn.CrossEntropyLoss(weight=class_weights)
loss_function_span2 = nn.CrossEntropyLoss(weight=class_weights)
loss_function_span3 = nn.CrossEntropyLoss(weight=class_weights)
loss_function_span4 = nn.CrossEntropyLoss(weight=class_weights)
optimizer     = optim.Adam(model.parameters(), lr=float(lr_str))

print('target data is ' + dataname, end = '\n\n')
print('batch size is {0}'.format(batch))
print('Number of train, develop, test is {0}, {1}, {2}'.format(train_size,devel_size,test_size))
print('learning rate is  {0}'.format(lr_str))
print('network strucuture is {0}'.format(network_structure))
print('iteration is {0}'.format(config.get('main', 'N_EPOCH')))
print('0_logit weight is {0}'.format(weight), end = '\n\n')

print('tensorboard writer logdir is {0}'.format(writer_log_dir))
print('brat logdir is {0}'.format(brat_log_dir))
print('Start Training... ',end ='\n\n')

start = time.time()
model.train()
#pdb.set_trace()

span1_max_f = -1
span2_max_f = -1
span3_max_f = -1
span4_max_f = -1

NER_max_scores = [span1_max_f, span2_max_f, span3_max_f, span4_max_f]

for epoch in range(int(config.get('main', 'N_EPOCH'))):
    # pdb.set_trace()
    # if epoch == 100:
    #     pdb.set_trace()
    print('Current Epoch:{}'.format(epoch+1))
    model.train()
    sum_loss_tr = 0.0
    for i, [n_doc, words, attention_mask, y_span_size_1, y_span_size_2, y_span_size_3, y_span_size_4] in enumerate(tqdm.tqdm(train_loader)):
        # pdb.set_trace()
        model.zero_grad()

        logits_span1, logits_span2, logits_span3, logits_span4 = model(words, attention_mask)
        loss_span1       = loss_function_span1(logits_span1, y_span_size_1)
        loss_span2       = loss_function_span2(logits_span2, y_span_size_2)
        loss_span3       = loss_function_span3(logits_span3, y_span_size_3)
        loss_span4       = loss_function_span4(logits_span4, y_span_size_4)

        loss = loss_span1 + loss_span2 + loss_span3 + loss_span4
        # loss = loss_span1
        loss.backward()
        optimizer.step()

    sum_loss = 0.0

    predicts_span1 = []
    answers_span1  = []

    predicts_span2 = []
    answers_span2  = []

    predicts_span3 = []
    answers_span3  = []

    predicts_span4 = []
    answers_span4  = []

    # predicts_RE  = []
    # answers_RE   = []

    # pdb.set_trace()
    for i, [n_doc, words, attention_mask, y_span_size_1, y_span_size_2, y_span_size_3, y_span_size_4] in enumerate(devel_loader):
        # pdb.set_trace()
        model.eval()
        batch_size = words.shape[0]

        logits_span1, logits_span2, logits_span3, logits_span4 = model(words, attention_mask)

        loss_span1       = loss_function_span1(logits_span1, y_span_size_1)
        loss_span2       = loss_function_span2(logits_span2, y_span_size_2)
        loss_span3       = loss_function_span3(logits_span3, y_span_size_3)
        loss_span4       = loss_function_span4(logits_span4, y_span_size_4)

        loss = loss_span1 + loss_span2 + loss_span3 + loss_span4
        sum_loss  += float(loss) * batch_size

        predicts_span1.append(torch.max(logits_span1, 1)[1])
        predicts_span2.append(torch.max(logits_span2, 1)[1])
        predicts_span3.append(torch.max(logits_span3, 1)[1])
        predicts_span4.append(torch.max(logits_span4, 1)[1])

        answers_span1.append(y_span_size_1)
        answers_span2.append(y_span_size_2)
        answers_span3.append(y_span_size_3)
        answers_span4.append(y_span_size_4)


############# index to texts ##########
    # sentences = [a for a in words]
    # predicted_tokens = []
    # for s_num in range(0, words.size(0)):
    #     one_predicted_tokens = []
    #     for i_num in range(0, words.size(1)):
    #         predicted_token = tokenizer.convert_ids_to_tokens(sentences[s_num][i_num].item())
    #         one_predicted_tokens.append(predicted_token)
    #     predicted_tokens.append(one_predicted_tokens)


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
        ##################################################################################

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


    writer.add_scalar('span1_devel/loss/epoch', sum_loss, epoch)
    writer.add_scalar('span1_devel/micro_precision/epoch', micro_p_r_f_span1[0],epoch)
    writer.add_scalar('span1_devel/micro_recall/epoch', micro_p_r_f_span1[1],epoch)
    writer.add_scalar('span1_devel/micro_f-measure/epoch', micro_p_r_f_span1[2],epoch)

    writer.add_scalar('span2_devel/loss/epoch', sum_loss, epoch)
    writer.add_scalar('span2_devel/micro_precision/epoch', micro_p_r_f_span2[0],epoch)
    writer.add_scalar('span2_devel/micro_recall/epoch', micro_p_r_f_span2[1],epoch)
    writer.add_scalar('span2_devel/micro_f-measure/epoch', micro_p_r_f_span2[2],epoch)

    writer.add_scalar('span3_devel/loss/epoch', sum_loss, epoch)
    writer.add_scalar('span3_devel/micro_precision/epoch', micro_p_r_f_span3[0],epoch)
    writer.add_scalar('span3_devel/micro_recall/epoch', micro_p_r_f_span3[1],epoch)
    writer.add_scalar('span3_devel/micro_f-measure/epoch', micro_p_r_f_span3[2],epoch)

    writer.add_scalar('span4_devel/loss/epoch', sum_loss, epoch)
    writer.add_scalar('span4_devel/micro_precision/epoch', micro_p_r_f_span4[0],epoch)
    writer.add_scalar('span4_devel/micro_recall/epoch', micro_p_r_f_span4[1],epoch)
    writer.add_scalar('span4_devel/micro_f-measure/epoch', micro_p_r_f_span4[2],epoch)


    if (epoch+1) % 100 == 0:
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
                logits_span1, logits_span2, logits_span3, logits_span4 = model(words, attention_mask)

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
            pred2ann.main(epoch+1, brat_log_dir, doc_correspnd_info_dict, n_docs, predicted_tokens, for_txt_span1, for_txt_span2, for_txt_span3, for_txt_span4)

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
