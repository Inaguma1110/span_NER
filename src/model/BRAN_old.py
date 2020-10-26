import itertools
import gensim
from collections import defaultdict
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sentencepiece as spm
from transformers import BertTokenizer, BertModel, BertForTokenClassification, BertForMaskedLM, BertConfig

import numpy as np

import pdb
import shelve

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PAIRS_MODULE():
    def __init__(self, batch_pred1, batch_pred2, batch_pred3, batch_pred4):
        self.number_batch = len(batch_pred1[0])
        self.batch_pred1 = batch_pred1[0].cpu().numpy()
        self.batch_pred2 = batch_pred2[0].cpu().numpy()
        self.batch_pred3 = batch_pred3[0].cpu().numpy()
        self.batch_pred4 = batch_pred4[0].cpu().numpy()
        self.batch_preds = [self.batch_pred1,self.batch_pred2,self.batch_pred3,self.batch_pred4]
        self.position_out = []
        self.pair_dict = defaultdict(list)

    def MAKE_POSITION_DATA(self):
        for pred_n, batch_pred in enumerate(self.batch_preds):
            for batch_number, batch in enumerate(batch_pred):
                for indx, logit in enumerate(batch):
                    if logit == 1:
                        self.position_out.append((batch_number, (pred_n+1, indx)))
        # return self.position_out
        for unit in self.position_out:
            self.pair_dict[unit[0]].append(unit[1])
        for num in range(self.number_batch):
            self.pair_dict.setdefault(num,[])
        # return self.pair_dict

    def MAKE_PAIR(self):
        # pdb.set_trace()
        self.MAKE_POSITION_DATA()
        for batch_number, unit in self.pair_dict.items():
            self.pair_list = []
            for pair_tuple in itertools.combinations(unit, 2):
                self.pair_list.append(list(pair_tuple))
            self.pair_dict[batch_number] = self.pair_list
        return dict(self.pair_dict)


class BERT_PRETRAINED_MODEL_JAPANESE(nn.Module):
    def __init__(self,config, vocab):
        super(BERT_PRETRAINED_MODEL_JAPANESE, self).__init__()
        self.config = config
        self.vocab = vocab
        self.BERT_config = BertConfig.from_json_file('../published_model/bert_spm/bert_config.json')
        self.tokenizer = BertTokenizer.from_pretrained('./spm_model/wiki-ja.vocab.txt')
        self.pretrained_BERT_model = BertModel.from_pretrained('../published_model/bert_spm/pytorch_model.bin',config=self.BERT_config)

    def return_tokenizer(self):
        return self.tokenizer

    def return_model(self):
        return self.pretrained_BERT_model

class SPAN_CNN(BERT_PRETRAINED_MODEL_JAPANESE):
    def __init__(self, config, vocab, REL_DIC):
        super(SPAN_CNN, self).__init__(config, vocab)
        #init defintion
        self.REL_DIC = REL_DIC
        # set numbers
        self.max_sent_len = int(config.get('makedata', 'MAX_SENT_LEN'))
        self.bert_pooling_layer = int(config.get('CNNs', 'BERT_POOLING_LAYER'))
        self.hidden_dim = int(config.get('CNNs', 'HIDDEN_DIM'))
        self.logit_h_dim = int(config.get('CNNs', 'LOGIT_H_DIM'))
        self.window_size = int(config.get('CNNs', 'WINDOW_SIZE'))
        self.embedding_dim = int(config.get('CNNs', 'BERT_INIT_EMBEDDING_DIM'))
        self.binary_span_dim = 2
        self.span1_kernel_size = 1
        self.span2_kernel_size = 2
        self.span3_kernel_size = 3
        self.span4_kernel_size = 4
        self.relsize = len(REL_DIC)

        self.input_liniar1_dim     = self.embedding_dim
        self.output_linear1_dim    = self.embedding_dim

        self.input_liniar2_dim     = self.hidden_dim
        self.output_linear2_dim    = self.binary_span_dim
        
        self.input_conv1_dim       = self.output_linear1_dim
        self.output_conv1_dim      = self.embedding_dim


        self.input_span1_dim       = self.embedding_dim
        self.output_span1_dim      = self.hidden_dim

        self.input_span2_dim       = self.embedding_dim
        self.output_span2_dim      = self.hidden_dim

        self.input_span3_dim       = self.embedding_dim
        self.output_span3_dim      = self.hidden_dim

        self.input_span4_dim       = self.embedding_dim
        self.output_span4_dim      = self.hidden_dim


        # set layers
        self.linear           = nn.Linear(self.input_liniar1_dim, self.output_linear1_dim)
        self.conv_1           = nn.Conv1d(self.input_conv1_dim, self.output_conv1_dim, self.window_size, padding=((self.window_size//2), ))
        # self.conv_2           = nn.Conv1d(self.input_conv2_dim, self.output_conv2_dim, self.window_size, padding=((self.window_size//2), ))
        # self.conv_3           = nn.Conv1d(self.input_conv3_dim, self.output_conv3_dim, self.window_size, padding=((self.window_size//2), ))
        # self.conv_4           = nn.Conv1d(self.input_conv4_dim, self.output_conv4_dim, self.window_size, padding=((self.window_size//2), ))
        self.span1_filter     = nn.Conv1d(self.input_span1_dim, self.output_span1_dim, self.span1_kernel_size )
        self.span2_filter     = nn.Conv1d(self.input_span2_dim, self.output_span2_dim, self.span2_kernel_size )
        self.span3_filter     = nn.Conv1d(self.input_span3_dim, self.output_span3_dim, self.span3_kernel_size )
        self.span4_filter     = nn.Conv1d(self.input_span4_dim, self.output_span4_dim, self.span4_kernel_size )        
        self.linear2_1          = nn.Linear(self.input_liniar2_dim, self.output_linear2_dim)
        self.linear2_2          = nn.Linear(self.input_liniar2_dim, self.output_linear2_dim)
        self.linear2_3          = nn.Linear(self.input_liniar2_dim, self.output_linear2_dim)
        self.linear2_4          = nn.Linear(self.input_liniar2_dim, self.output_linear2_dim)
        self.span1_linear     = nn.Linear(self.embedding_dim, self.binary_span_dim)
        self.span2_linear     = nn.Linear(self.embedding_dim, self.binary_span_dim)
        self.span3_linear     = nn.Linear(self.embedding_dim, self.binary_span_dim)
        self.span4_linear     = nn.Linear(self.embedding_dim, self.binary_span_dim)

        self.dropout      = nn.Dropout(p=0.5)
        self.softmax      = nn.Softmax(dim =self.binary_span_dim)

    def SharePart(self, tokens_tensor, attention_mask):
        # self.pretrained_BERT_model.eval()
        with torch.no_grad():
            all_encoder_layers = self.pretrained_BERT_model(tokens_tensor, attention_mask=attention_mask)
        rep_word      = all_encoder_layers[self.bert_pooling_layer]
        rep_word = rep_word * attention_mask.view(-1,150,1)
        ###  SHARE PART ###
        input_dropout = self.dropout(rep_word)
        h_fc          = self.linear(input_dropout)
        foo_vecs      = F.relu(h_fc).permute(0,2,1)
        h_conv        = F.relu(self.conv_1(foo_vecs))
        return h_conv

    def forward(self, tokens_tensor, attention_mask, relation_flag, pred_pair_unit):
        # pdb.set_trace()
        h_conv1 = self.SharePart(tokens_tensor,attention_mask)

        logits_span_1 = self.linear2_1(F.relu(self.span1_filter(h_conv1).permute(0,2,1))).permute(0,2,1)
        logits_span_2 = self.linear2_2(F.relu(self.span2_filter(F.pad(h_conv1, (0,self.span2_kernel_size - 1))).permute(0,2,1))).permute(0,2,1)
        logits_span_3 = self.linear2_3(F.relu(self.span3_filter(F.pad(h_conv1, (0,self.span3_kernel_size - 1))).permute(0,2,1))).permute(0,2,1)
        logits_span_4 = self.linear2_4(F.relu(self.span4_filter(F.pad(h_conv1, (0,self.span4_kernel_size - 1))).permute(0,2,1))).permute(0,2,1)

        return logits_span_1, logits_span_2, logits_span_3, logits_span_4

class RELATION(SPAN_CNN):
    def __init__(self, config, vocab, REL_DIC):
        # pdb.set_trace()
        super(RELATION, self).__init__(config, vocab, REL_DIC)
        # self.rel_size = len(self.REL_DIC)
        self.rel_size = 2
        self.hidden_dim = int(config.get('CNNs', 'HIDDEN_DIM'))
        # self.hidden_dim = 1000
        self.rel_conv         = nn.Conv1d(self.output_conv1_dim+2, self.hidden_dim, self.window_size, padding=((self.window_size//2), ))
        self.rel_linear1      = nn.Linear(self.max_sent_len, 1)
        self.rel_linear2      = nn.Linear(self.hidden_dim, 2)
        self.encoder_layer1   = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer1, num_layers=6)


    def forward(self, rel_word_x, rel_attention_mask, pred_pair_unit, dflag):
        h_conv2 = self.SharePart(rel_word_x, rel_attention_mask)
        entity1_position_vec = torch.zeros(len(pred_pair_unit),1,self.max_sent_len).to('cuda:0' if torch.cuda.is_available() else 'cpu')
        entity2_position_vec = torch.zeros(len(pred_pair_unit),1,self.max_sent_len).to('cuda:0' if torch.cuda.is_available() else 'cpu')
        # pdb.set_trace()
        for m, mini_batch in enumerate(pred_pair_unit):
            for i, unit in enumerate(mini_batch):
                entity_structure_num = unit[0]
                entity_start_index = unit[1]
                for j in range(entity_start_index, entity_start_index+entity_structure_num-1):
                    try:
                        if i%2==0:
                            entity1_position_vec[m][0][j] = 1
                        if i%2==1:
                            entity2_position_vec[m][0][j] = 1
                    except:
                        pass
        h_conv1_entity = torch.cat([h_conv2, entity1_position_vec, entity2_position_vec],dim=1)
        
        # pdb.set_trace()
        rel = self.rel_conv(h_conv1_entity)
        trans = self.transformer_encoder(rel.permute(0,2,1))
        rel_rep  = self.rel_linear1(trans.permute(0,2,1))
        rel_logits = self.rel_linear2(rel_rep.permute(0,2,1)).squeeze(dim=1)#error

        return rel_logits


class A():
    pass

class B():
    pass

class C:
    def forward(self, text):
        # encode text
        text_encoded = Bert(text)
        # NER with Class A
        ner_tag = A(text_encoded)
        # RE with class B
        re_tag = B(text_encoded,ner_tag)













class BRAN(SPAN_CNN):
    def __init__(self, config, vocab, REL_DIC):
        # pdb.set_trace()
        super(BRAN, self).__init__(config, vocab, REL_DIC)
        self.rel_size = len(self.REL_DIC)
        self.hidden_dim = int(config.get('CNNs', 'HIDDEN_DIM'))

        self.rel_conv_head         = nn.Conv1d(self.output_conv1_dim+2, self.hidden_dim, self.window_size, padding=((self.window_size//2), ))
        self.rel_conv_tail         = nn.Conv1d(self.output_conv1_dim+2, self.hidden_dim, self.window_size, padding=((self.window_size//2), ))

        self.rel_linear1      = nn.Linear(self.max_sent_len, 1)

        self.rel_linear_head      = nn.Linear(self.hidden_dim, 6)
        self.rel_linear_tail      = nn.Linear(self.hidden_dim, 6)
        
        self.affine1          = nn.Bilinear(self.max_sent_len,self.max_sent_len, self.max_sent_len)

    def forward(self, rel_word_x, rel_attention_mask, pred_pair_unit):
        h_conv2 = self.SharePart(rel_word_x, rel_attention_mask)

        entity1_position_vec = torch.zeros(len(pred_pair_unit),1,self.max_sent_len).to('cuda:0' if torch.cuda.is_available() else 'cpu')
        entity2_position_vec = torch.zeros(len(pred_pair_unit),1,self.max_sent_len).to('cuda:0' if torch.cuda.is_available() else 'cpu')
        # pdb.set_trace()
        for m, mini_batch in enumerate(pred_pair_unit):
            for i, unit in enumerate(mini_batch):
                entity_structure_num = unit[0]
                entity_start_index = unit[1]
                for j in range(entity_start_index, entity_start_index+entity_structure_num-1):
                    try:
                        if i%2==0:
                            entity1_position_vec[m][0][j] = 1
                        if i%2==1:
                            entity2_position_vec[m][0][j] = 1
                    except:
                        pass

        h_conv1_entity = torch.cat([h_conv2, entity1_position_vec, entity2_position_vec],dim=1)
        rep_rel_plus = torch.unsqueeze(h_conv1_entity,2)

        rel1 = self.rel_conv_head(h_conv1_entity)
        rel_flat1 = rel1.squeeze()
        rel_head_rep = self.rel_linear_head(rel_flat1.permute(0,2,1))

        rel2 = self.rel_conv_tail(h_conv1_entity)
        rel_flat2 = rel2.squeeze()
        rel_tail_rep = self.rel_linear_tail(rel_flat2.permute(0,2,1))
        
        rel_rep  = self.affine1(rel_flat1,rel_flat2)
        rel_logits = self.rel_linear2(rel_rep.permute(0,2,1)).squeeze()

        return rel_logits
