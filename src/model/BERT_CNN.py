import gensim
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sentencepiece as spm
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

import numpy as np

import pdb
import shelve

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        self.output_linear1_dim    = self.hidden_dim


        
        # self.input_conv1_dim       = self.output_linear1_dim
        # self.output_conv1_dim      = self.embedding_dim

        # self.input_conv2_dim       = self.output_conv1_dim
        # self.output_conv2_dim      = self.embedding_dim
        
        # self.input_conv3_dim       = self.output_conv2_dim
        # self.output_conv3_dim      = self.embedding_dim

        # self.input_conv4_dim       = self.output_conv3_dim
        # self.output_conv4_dim      = self.embedding_dim

        self.input_liniar2_dim     = self.hidden_dim
        self.output_linear2_dim    = self.binary_span_dim


        # set layers
        self.linear           = nn.Linear(self.input_liniar1_dim, self.output_linear1_dim)
        # self.conv_1           = nn.Conv1d(self.input_conv1_dim, self.output_conv1_dim, self.window_size, padding=((self.window_size//2), ))
        # # self.conv_2           = nn.Conv1d(self.input_conv2_dim, self.output_conv2_dim, self.window_size, padding=((self.window_size//2), ))
        # # self.conv_3           = nn.Conv1d(self.input_conv3_dim, self.output_conv3_dim, self.window_size, padding=((self.window_size//2), ))
        # # self.conv_4           = nn.Conv1d(self.input_conv4_dim, self.output_conv4_dim, self.window_size, padding=((self.window_size//2), ))
        self.linearspan1          = nn.Linear(self.input_liniar2_dim, self.output_linear2_dim)
        self.linearspan2          = nn.Linear(self.input_liniar2_dim, self.output_linear2_dim)
        self.linearspan3          = nn.Linear(self.input_liniar2_dim, self.output_linear2_dim)
        self.linearspan4          = nn.Linear(self.input_liniar2_dim, self.output_linear2_dim)
        # self.span1_filter     = nn.Conv1d(self.embedding_dim, self.hidden_dim, self.span1_kernel_size, padding=((self.span1_kernel_size//2), ))
        # self.span2_filter     = nn.Conv1d(self.embedding_dim, self.hidden_dim, self.span2_kernel_size )
        # self.span3_filter     = nn.Conv1d(self.embedding_dim, self.hidden_dim, self.span3_kernel_size, padding=((self.span3_kernel_size//2), ))
        # self.span4_filter     = nn.Conv1d(self.embedding_dim, self.hidden_dim, self.span4_kernel_size )        
        self.dropout      = nn.Dropout(p=0.5)

    def forward(self, tokens_tensor):
        # pdb.set_trace()
        with torch.no_grad():
            all_encoder_layers = self.pretrained_BERT_model(tokens_tensor)
        rep_word      = all_encoder_layers[self.bert_pooling_layer]

        ###  SHARE PART ###
        input_dropout = self.dropout(rep_word)
        h_fc          = self.linear(input_dropout)
        # foo_vecs      = h_fc.permute(0,2,1)
        # h_conv1       = self.conv_1(foo_vecs)
        # h_conv2       = self.conv_2(h_conv1)
        # h_conv3       = self.conv_3(h_conv2)
        # h_conv4       = self.conv_4(h_conv3)


        ###  NER PART ###
        # logits_span_1 = self.span1_filter(h_conv1).squeeze()
        # logits_span_2 = self.span2_filter(F.pad(h_conv1, (0,self.span2_kernel_size - 1))).squeeze()
        # logits_span_3 = self.span3_filter(h_conv1).squeeze()
        # logits_span_4 = self.span4_filter(F.pad(h_conv1, (0,self.span4_kernel_size - 1))).squeeze()


        logits_span_1 = self.linearspan1(h_fc).permute(0,2,1)
        logits_span_2 = self.linearspan2(h_fc).permute(0,2,1)
        logits_span_3 = self.linearspan3(h_fc).permute(0,2,1)
        logits_span_4 = self.linearspan4(h_fc).permute(0,2,1)


        return logits_span_1, logits_span_2, logits_span_3, logits_span_4

