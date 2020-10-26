import sys

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as D

import numpy as np

import argparse
import pdb
import tqdm, contextlib
import time
import shelve
import configparser
from distutils.util import strtobool

from BIO_CNN import BERT_PRETRAINED_MODEL_JAPANESE, BIO_NER_MODEL
#import score
import pred2sent
import tensorboardX as tb

config = configparser.ConfigParser()
config.read('../machine.conf')

dataname          = config.get('dataname', 'BIO')### 'data' or 'Machining_data' or 'Machining_data_kakogijutsu'
max_sent_len      = int(config.get('makedata', 'MAX_SENT_LEN'))
network_structure = config.get('CNNs', 'NETWORK_STRUCTURE')
weight            = config.get('CNNs', 'WEIGHT')
batch             = config.get('main', 'BATCH_SIZE_TRAIN')
lr_str            = config.get('main', 'LEARNING_RATE')
attention_mask_is = config.get('CNNs', 'ATTENTION_MASK_IS')

writer_log_dir='../../data/TensorboardGraph/BIO_NER_RE/batch_size_{0}/learning_rate_{1}/network_{2}_{3}/0_logit_weight_{4}'.format(batch,lr_str,network_structure,attention_mask_is,weight)
hoge_dir = '../../data/TensorboardGraph/BIO_NER_RE/debug'

np.random.seed(1)
torch.manual_seed(1)
# pdb.set_trace()
writer = tb.SummaryWriter(logdir = writer_log_dir)


print('\nCreate Environment...\n')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


print('\nCreate data...')
database = shelve.open(config.get('path', 'SHELVE_PATH'))
vocab, corpus, TAG_DIC, filename_lst = database[dataname] 
database.close()

document            = [a[0] for a in corpus]
word_input          = torch.LongTensor([a[1] for a in corpus]).to(device)
y_BIO_map           = torch.LongTensor([a[2] for a in corpus]).to(device) 
attention_mask      = torch.LongTensor([a[3] for a in corpus]).to(device)
sentencepieced      = [a[4] for a in corpus]
removed_Entdic      = [a[5] for a in corpus]
Reldic              = [a[6] for a in corpus]

dataset = D.TensorDataset(word_input, attention_mask, y_BIO_map)

train_size = int(0.8 * len(word_input))
devel_size = int(0.1 * len(word_input))
test_size = len(word_input) - train_size - devel_size

train_dataset, devel_dataset, test_dataset = D.random_split(dataset, [train_size, devel_size, test_size])
train_loader = D.DataLoader(train_dataset, batch_size=int(config.get('main', 'BATCH_SIZE_TRAIN')), shuffle=strtobool(config.get('main', 'BATCH_SHUFFLE_TRAIN')))
devel_loader = D.DataLoader(devel_dataset, batch_size=int(config.get('main', 'BATCH_SIZE_DEVEL')), shuffle=strtobool(config.get('main', 'BATCH_SHUFFLE_DEVEL')))
test_loader = D.DataLoader(test_dataset, batch_size=int(config.get('main', 'BATCH_SIZE_TEST')), shuffle=strtobool(config.get('main', 'BATCH_SHUFFLE_TEST')))

print('finish', end='\n')

print('Create Model...')

bert_pretrained_model_japanese = BERT_PRETRAINED_MODEL_JAPANESE(config, vocab)
tokenizer = bert_pretrained_model_japanese.return_tokenizer()
pretrained_BERT_model = bert_pretrained_model_japanese.return_model()

model  = BIO_NER_MODEL(config, vocab).to(device)


print('finish', end='\n\n')

weights = [float(s) for s in config.get('CNNs', 'LOSS_WEIGHT').split(',')]
class_weights = torch.FloatTensor(weights).to(device)

loss_function = nn.CrossEntropyLoss(weight=None)

# loss_function_re   = nn.CrossEntropyLoss()

optimizer     = optim.Adam(model.parameters(), lr=float(lr_str))
#optimizer_re  = optim.Adam(model_re.parameters(), lr=float(config.get('main', 'LEARNING_RATE')))

print('target data is ' + dataname, end = '\n\n')
print('batch size is {0}'.format(batch))
print('learning rate is  {0}'.format(lr_str))
print('network strucuture is {0}'.format(network_structure))
print('0_logit weight is {0}'.format(weight), end = '\n\n')
print('tensorboard writer logdir is {0}'.format(writer_log_dir))

print('Start Training... ',end ='\n\n')

start = time.time()
model.train()
#pdb.set_trace()

max_f = -1

for epoch in range(int(config.get('main', 'N_EPOCH'))):
    # pdb.set_trace()
    # if epoch == 100:
    #     pdb.set_trace()
    print('Current Epoch:{}'.format(epoch+1))
    model.train()
    sum_loss_tr = 0.0

    for i, [words, attention_mask, y_BIO_map] in enumerate(tqdm.tqdm(train_loader)):
        # pdb.set_trace()
        model.zero_grad()

        logits = model(words, attention_mask)
        loss      = loss_function(logits, y_BIO_map)
        loss.backward()
        optimizer.step()

    print(sum_loss_tr)
    sum_loss = 0.0
    predicts = []
    answers  = []
    for i, [words, attention_mask, y_BIO_map] in enumerate(devel_loader):
        # pdb.set_trace()
        model.eval()
        batch_size = words.shape[0]
        logits = model(words, attention_mask)
        loss       = loss_function(logits, y_BIO_map)

        sum_loss  += float(loss) * batch_size

        predicts.append(torch.max(logits, 1)[1])
        answers.append(y_BIO_map)


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
    print('calculate ner score ...')
    preds = torch.cat(predicts, 0).view(-1,1).squeeze().cpu().numpy()
    golds = torch.cat(answers, 0).view(-1,1).squeeze().cpu().numpy()
    micro_p_r_f = precision_recall_fscore_support(golds, preds, labels=[1,2,3], average='micro')    
    acc_span1 = accuracy_score(golds,preds)


    print('micro p/r/F score is ' + str(micro_p_r_f))
    print('acc score is ' + str(acc_span1))

    if max_f < micro_p_r_f[2]:
        max_scores = (epoch,micro_p_r_f)
        max_f = micro_p_r_f[2]


    writer.add_scalar('BIO_devel/loss/epoch', sum_loss, epoch)
    writer.add_scalar('BIO_devel/micro_precision/epoch', micro_p_r_f[0],epoch)
    writer.add_scalar('BIO_devel/micro_recall/epoch', micro_p_r_f[1],epoch)
    writer.add_scalar('BIO_devel/micro_f-measure/epoch', micro_p_r_f[2],epoch)

    if (epoch+1) % 100 == 0:
        model.eval()

        predicts = []
        answers  = []


        answers_token_preds_golds = []

        with torch.no_grad():
            for i, [words, attention_mask, y_BIO_map] in enumerate(test_loader):
                logits = model(words, attention_mask)

                predicts.append(torch.max(logits, 1)[1])

                answers.append(y_BIO_map)

                # pdb.set_trace()
                sentences = [a for a in words]
                predicted_tokens = []

                for s_num in range(0, words.size(0)):
                    one_predicted_tokens = []
                    for i_num in range(0, words.size(1)):
                        predicted_token = tokenizer.convert_ids_to_tokens(sentences[s_num][i_num].item())
                        one_predicted_tokens.append(predicted_token)
                    predicted_tokens.append(one_predicted_tokens)

                for num_unit in range(0, words.size(0)):
                    answers_token_preds_golds.append((predicted_tokens[num_unit], predicts[0][num_unit], answers[0][num_unit]))

                # pdb.set_trace()


            print('Test  ###############################################################　calculate NER score ...  ###########################################################')
            print('calculate NER score ...')


            preds = torch.cat(predicts, 0).view(-1,1).squeeze().cpu().numpy()
            golds = torch.cat(answers, 0).view(-1,1).squeeze().cpu().numpy()
            micro_p_r_f = precision_recall_fscore_support(golds,preds,labels=[1,2,3], average='micro')    


            print('BIO micro p/r/F score is ' + str(micro_p_r_f))

        model.train()

print('span max score is ' + str(max_f))
# print('RE max score is ' + str(RE_max_scores))