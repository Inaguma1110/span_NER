import itertools
import gensim
from collections import defaultdict
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sentencepiece as spm
from pytorch_transformers import BertTokenizer, BertModel, BertForTokenClassification, BertForMaskedLM, BertConfig

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
        # pdb.set_trace()
        for pred_n, batch_pred in enumerate(self.batch_preds):
            for batch_number, batch in enumerate(batch_pred):
                for indx, logit in enumerate(batch):
                    if logit == 1:
                        self.position_out.append((batch_number, (pred_n+1, indx)))
        # return self.position_out
        for unit in self.position_out:
            self.pair_dict[unit[0]].append(unit[1])
        # pdb.set_trace()
        for num in range(self.number_batch-1):
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
    def __init__(self, config, vocab, REL_DIC, REL_LABEL_DICT):
        super(SPAN_CNN, self).__init__(config, vocab)

        #init defintion
        self.REL_DIC = REL_DIC
        self.REL_LABEL_DICT = REL_LABEL_DICT

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
        self.linear2          = nn.Linear(self.input_liniar2_dim, self.output_linear2_dim)
        
        self.span1_linear     = nn.Linear(self.embedding_dim, self.binary_span_dim)
        self.span2_linear     = nn.Linear(self.embedding_dim, self.binary_span_dim)
        self.span3_linear     = nn.Linear(self.embedding_dim, self.binary_span_dim)
        self.span4_linear     = nn.Linear(self.embedding_dim, self.binary_span_dim)

        self.dropout      = nn.Dropout(p=0.5)
        self.softmax      = nn.Softmax(dim =self.binary_span_dim)

    def forward(self, tokens_tensor, attention_mask, relation_flag, pred_pair_unit, n_doc):
       # pdb.set_trace()
        # self.pretrained_BERT_model.eval()
        with torch.no_grad():
            all_encoder_layers = self.pretrained_BERT_model(tokens_tensor, attention_mask=attention_mask)
        rep_word      = all_encoder_layers[self.bert_pooling_layer]
        
        rep_word * attention_mask.view(-1,150,1)
        attention_masks  = torch.stack([attention_mask, attention_mask], dim = 1)
        
        # logits_span_1 = self.span1_linear(rep_word).permute(0,2,1)
        # logits_span_2 = self.span2_linear(rep_word).permute(0,2,1)
        # logits_span_3 = self.span3_linear(rep_word).permute(0,2,1)
        # logits_span_4 = self.span4_linear(rep_word).permute(0,2,1)

        ###  SHARE PART ###
        input_dropout = self.dropout(rep_word)
        h_fc          = self.linear(input_dropout)
        foo_vecs      = h_fc.permute(0,2,1)
        h_conv1       = self.conv_1(foo_vecs)


        logits_span_1 = self.linear2(self.span1_filter(h_conv1).permute(0,2,1)).permute(0,2,1)
        logits_span_2 = self.linear2(self.span2_filter(F.pad(h_conv1, (0,self.span2_kernel_size - 1))).permute(0,2,1)).permute(0,2,1)
        logits_span_3 = self.linear2(self.span3_filter(F.pad(h_conv1, (0,self.span3_kernel_size - 1))).permute(0,2,1)).permute(0,2,1)
        logits_span_4 = self.linear2(self.span4_filter(F.pad(h_conv1, (0,self.span4_kernel_size - 1))).permute(0,2,1)).permute(0,2,1)

        return h_conv1, logits_span_1, logits_span_2, logits_span_3, logits_span_4

        # if relation_flag == False:
        #     return logits_span_1, logits_span_2, logits_span_3, logits_span_4

        # if relation_flag == True:
        #     return input_dropout


class RELATION(SPAN_CNN):
    def __init__(self, config, vocab, REL_DIC, REL_LABEL_DICT):
        # pdb.set_trace()
        super(RELATION, self).__init__(config, vocab, REL_DIC, REL_LABEL_DICT)
        self.rel_size = len(self.REL_DIC)
        self.hidden_dim = int(config.get('CNNs', 'HIDDEN_DIM'))
        self.rel_conv         = nn.Conv1d(self.output_conv1_dim+2, self.hidden_dim, self.window_size, padding=((self.window_size//2), ))
        self.rel_linear1      = nn.Linear(self.max_sent_len, 1)
        self.rel_linear2      = nn.Linear(self.hidden_dim, 6)

    def forward(self, h_conv, pred_pair_unit):
        entity1_position_vec = torch.zeros(1,1,150).to('cuda:0' if torch.cuda.is_available() else 'cpu')
        entity2_position_vec = torch.zeros(1,1,150).to('cuda:0' if torch.cuda.is_available() else 'cpu')
        for i, unit in enumerate(pred_pair_unit):
            entity_structure_num = unit[0]
            entity_start_index = unit[1]
            for j in range(entity_start_index, entity_start_index+entity_structure_num-1):
                try:
                    if i%2==0:
                        entity1_position_vec[0][0][j] = 1
                    if i%2==1:
                        entity2_position_vec[0][0][j] = 1
                except:
                    pass

        h_conv1 = h_conv
        h_conv1_entity = torch.cat([h_conv1, entity1_position_vec, entity2_position_vec],dim=1)
        rel = self.rel_conv(h_conv1_entity)
        rel_flat = rel.squeeze()
        rel_rep  = self.rel_linear1(rel_flat)
        rel_logits = self.rel_linear2(rel_rep.permute(1,0))

        return rel_logits
