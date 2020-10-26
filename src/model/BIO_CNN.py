import gensim
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


class BIO_NER_MODEL(BERT_PRETRAINED_MODEL_JAPANESE):
    def __init__(self, config, vocab):
        super(BIO_NER_MODEL, self).__init__(config, vocab)

        #init defintion
        # set numbers
        self.max_sent_len = int(config.get('makedata', 'MAX_SENT_LEN'))
        self.bert_pooling_layer = int(config.get('CNNs', 'BERT_POOLING_LAYER'))

        self.hidden_dim = int(config.get('CNNs', 'HIDDEN_DIM'))
        self.logit_h_dim = int(config.get('CNNs', 'LOGIT_H_DIM'))
        self.window_size = int(config.get('CNNs', 'WINDOW_SIZE'))
        self.embedding_dim = int(config.get('CNNs', 'BERT_INIT_EMBEDDING_DIM'))
        self.binary_span_dim = 2


        self.input_liniar1_dim     = self.embedding_dim
        self.output_linear1_dim    = 4  # B I O の３次元

        self.input_liniar2_dim     = self.hidden_dim
        self.output_linear2_dim    = self.binary_span_dim
        
        self.input_conv1_dim       = self.output_linear1_dim
        self.output_conv1_dim      = self.embedding_dim


        # set layers
        self.linear1           = nn.Linear(self.input_liniar1_dim, self.output_linear1_dim)
        self.conv_1            = nn.Conv1d(self.input_conv1_dim, self.output_conv1_dim, self.window_size, padding=((self.window_size//2), ))

        self.dropout      = nn.Dropout(p=0.5)
        # self.softmax      = nn.Softmax(dim =self.binary_span_dim)

    def forward(self, tokens_tensor, attention_mask):
        # pdb.set_trace()
        # self.pretrained_BERT_model.eval()
        with torch.no_grad():
            all_encoder_layers = self.pretrained_BERT_model(tokens_tensor, attention_mask=attention_mask)
        rep_word      = all_encoder_layers[self.bert_pooling_layer]
        logits = self.linear1(rep_word).permute(0,2,1)

        attention_masks  = torch.stack([attention_mask, attention_mask, attention_mask, attention_mask], dim = 1)
        mask_logits = logits * attention_masks
        return mask_logits

        # ###  SHARE PART ###
        # input_dropout = self.dropout(rep_word)
        # h_fc          = self.linear(input_dropout)
        # foo_vecs      = h_fc.permute(0,2,1)
        # h_conv1       = self.conv_1(foo_vecs)
        # # h_conv2       = self.conv_2(h_conv1)
        # # h_conv3       = self.conv_3(h_conv2)
        # # h_conv4       = self.conv_4(h_conv3)

        # logits = self.linear2(self.span1_filter(h_conv1).squeeze().permute(0,2,1)).permute(0,2,1)

        # return logits

