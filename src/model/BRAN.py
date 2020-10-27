import itertools
import random
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

def nest_cut(spans, span_size): #spans = (batch, span_size, seq_len)
    for m in range(len(spans)):
        for s in reversed(range(len(spans[m]))):
            for index in range(len(spans[m][s])):
                if spans[m][s][index] == 1:
                    for i in range(span_size):
                        if s-i-1>=0:
                            spans[m][s-i-1][index:index+s+1] = 0
    return spans

class PairsModule():
    def __init__(self, spans):
        self.number_batch = len(spans)
        self.spans = spans.permute(1,0,2).cpu().numpy()
        self.position_out = []
        self.pair_dict = defaultdict(list)

    def make_entity_position(self):
        for pred_n, batch_pred in enumerate(self.spans):
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

    def make_pair(self):
        self.make_entity_position()
        for batch_number, unit in self.pair_dict.items():
            self.pair_list = []
            for pair_tuple in itertools.combinations(unit, 2):
                self.pair_list.append(list(pair_tuple))
            self.pair_dict[batch_number] = self.pair_list
        return dict(self.pair_dict)


class JapaneseBertPretrainedModel(nn.Module):
    def __init__(self,config, vocab):
        super(JapaneseBertPretrainedModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.BERT_config = BertConfig.from_json_file('../published_model/bert_spm/bert_config.json')
        self.tokenizer = BertTokenizer.from_pretrained('./spm_model/wiki-ja.vocab.txt')
        self.pretrained_BERT_model = BertModel.from_pretrained('../published_model/bert_spm/pytorch_model.bin',config=self.BERT_config)
    def forward(self, *args, **kwargs):
        return self.pretrained_BERT_model(*args, **kwargs)

class SpanHead(nn.Module):
    def __init__(self, embedding_dim, window_size, hidden_dim, span_size):
        super(SpanHead, self).__init__()
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.span_size = span_size

        self.conv = nn.Conv1d(self.hidden_dim, self.hidden_dim, self.window_size, padding=((self.window_size//2), ))
        self.span_filter   = nn.Conv1d(self.embedding_dim, self.hidden_dim, self.span_size)
        self.linear        = nn.Linear(self.hidden_dim, 2)

    def forward(self, inputs):
        paded       = F.pad(inputs, (0,self.span_size-1))        # [B, 768, seq_len + span_size]
        spaned      = F.relu(self.span_filter(paded))            # [B, D, seq_len]
        spaned      = F.relu(self.conv(spaned))

        # share linear
        return spaned

        # each linear
        # lineard     = F.relu(self.linear(spaned.permute(0,2,1))) # [B, D, 2]
        # logits_span = lineard.permute(0,2,1)                     # [B, seq_len, 2]
        # return logits_span

class NamedEntityExtractionModule(nn.Module):
    def __init__(self,config, span_size):
        super(NamedEntityExtractionModule, self).__init__()
        self.span_size = span_size
        self.embedding_dim  = int(config.get('CNNs', 'BERT_INIT_EMBEDDING_DIM'))
        self.window_size    = int(config.get('CNNs', 'WINDOW_SIZE'))
        self.hidden_dim     = int(config.get('CNNs', 'HIDDEN_DIM'))
        self.heads = nn.ModuleList([SpanHead(self.embedding_dim, self.window_size, self.hidden_dim, span_size) for span_size in range(1, span_size+1)])

        self.share_linear = nn.Linear(self.hidden_dim, 2)
    def forward(self, tokens):
        # share linear
        spans = [self.heads[s_x](tokens) for s_x in range(self.span_size)]
        spans = torch.stack(spans, dim = 0)
        lineared = self.share_linear(spans.permute(0,1,3,2))
        logits_spans = lineared.permute(0,1,3,2)

        # each linear
        # logits_spans = [self.heads[s_x](tokens) for s_x in range(self.span_size)]
        # logits_spans = torch.stack(logits_spans, dim = 0)

        return logits_spans



class RelationExtractionModule(nn.Module):
    def __init__(self, config, vocab, rel_dic):
        super(RelationExtractionModule, self).__init__()
        self.rel_size = len(rel_dic)
        # self.rel_size = 2

        self.hidden_dim       = int(config.get('CNNs', 'HIDDEN_DIM'))
        self.embedding_dim    = int(config.get('CNNs', 'BERT_INIT_EMBEDDING_DIM'))
        self.window_size      = int(config.get('CNNs', 'WINDOW_SIZE'))
        self.max_sent_len     = int(config.get('makedata', 'MAX_SENT_LEN'))


        self.rel_conv         = nn.Conv1d(self.embedding_dim, self.embedding_dim, self.window_size, padding=((self.window_size//2), ))
        self.rel_conv2        = nn.Conv1d(self.embedding_dim, self.embedding_dim, self.window_size, padding=((self.window_size//2), ))

        # self.rel_linear      = nn.Linear(self.hidden_dim, self.rel_size)
        self.rel_linear       = nn.Linear(self.hidden_dim, self.rel_size)
        self.linear1          = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.linear2          = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear0          = nn.Linear(self.embedding_dim,self.embedding_dim)

        self.Pooling_1        = nn.MaxPool1d(4, stride=2)
        self.Pooling_2        = nn.MaxPool1d(4, stride=2)


    def make_entity_position(self, pred_pair_unit):
        entity1_position_vec = torch.zeros(len(pred_pair_unit),1,self.max_sent_len)
        entity2_position_vec = torch.zeros(len(pred_pair_unit),1,self.max_sent_len)
        # pdb.set_trace() #TODO
        for m, mini_batch in enumerate(pred_pair_unit):
            for i, unit in enumerate(mini_batch):
                entity_structure_num = unit[0]
                entity_start_index = unit[1]
                if i%2==0:
                    entity1_position_vec[m,0,entity_start_index:entity_structure_num+entity_start_index] = 1
                if i%2==1:
                    entity2_position_vec[m,0,entity_start_index:entity_structure_num+entity_start_index] = 1
        return entity1_position_vec, entity2_position_vec


    def forward(self, rel_word_x, data_unit_for_relation, dflag):
        rel_logits = []
        rel_labels = []
        rel_word_x = F.dropout(rel_word_x, 0.2) #[num_pair, D, seq_len]
        rel_word_x = F.relu(self.rel_conv(rel_word_x)) + rel_word_x #[num_pair, D, seq_len]
        rel_word_x = F.relu(self.rel_conv2(rel_word_x)) + rel_word_x #[num_pair, D, seq_len]

        for b, pair_list in data_unit_for_relation.items():
            if len(pair_list) > 0:
                for one_pair in pair_list:
                    head_span_size, head_index = one_pair[0][0]
                    tail_span_size, tail_index = one_pair[0][1]
                    rel_label = torch.from_numpy(np.array(one_pair[1],dtype=np.int64)).to(device)

                    # head_position_vec = torch.zeros(1,self.max_sent_len).to(rel_word_x)
                    # tail_position_vec = torch.zeros(1,self.max_sent_len).to(rel_word_x)

                    # head_position_vec[0,head_index:head_index+head_span_size] = 1
                    # tail_position_vec[0,tail_index:tail_index+tail_span_size] = 1

                    headrep = rel_word_x[b].permute(1,0)[head_index:head_index+head_span_size].max(0)[0]
                    tailrep = rel_word_x[b].permute(1,0)[tail_index:tail_index+tail_span_size].max(0)[0]

                    maxed = headrep+tailrep
                    maxed = F.relu(self.linear1(maxed))
                    maxed = F.relu(self.linear2(maxed))

                    maxed = F.dropout(maxed,0.5)
                    rel_logit = self.rel_linear(maxed)
                    rel_logits.append(rel_logit)
                    rel_labels.append(rel_label)
            else:
                pass
        # rel_word_x = F.dropout(self.linear0(rel_word_x.transpose(-1,-2)),0.2).transpose(-1,-2)
        rel_logits = torch.stack(tuple(rel_logits),dim=0)
        rel_labels = torch.stack(tuple(rel_labels),dim=0)
        return rel_logits, rel_labels



class MyModel(nn.Module):
    def __init__(self, span_size, config, vocab, rel_dic, REL_LABEL_DICT,doc_correspnd_info_dict):
        super(MyModel, self).__init__()

        self.bert_model = JapaneseBertPretrainedModel(config, vocab)
        self.span_size = span_size
        self.rel_dic = rel_dic
        self.REL_LABEL_DICT = REL_LABEL_DICT
        self.doc_correspnd_info_dict = doc_correspnd_info_dict
        self.NerModel = NamedEntityExtractionModule(config, span_size)
        self.ReModel = RelationExtractionModule(config, vocab, rel_dic)
        self.bert_pooling_layer = int(config.get('CNNs', 'BERT_POOLING_LAYER'))
        self.embedding_dim = int(config.get('CNNs', 'BERT_INIT_EMBEDDING_DIM'))
        self.window_size = int(config.get('CNNs', 'WINDOW_SIZE'))
        self.down_sampling_rate = float(config.get('main','DOWN_SAMPLING_RATE'))
        print("\ndown sampling rate is {0}".format(self.down_sampling_rate),end="\n")

        self.linear           = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.conv_1           = nn.Conv1d(self.embedding_dim, self.embedding_dim, self.window_size, padding=((self.window_size//2), ))
        self.dropout      = nn.Dropout(p=0.5)
        self.softmax      = nn.Softmax(dim = 2)

    def make_pair(self, spans, n_doc, Relation_gold_learning_switch):
        n_doc = n_doc.to('cpu').detach().numpy().copy().squeeze()

        if not Relation_gold_learning_switch: # NERの結果を使ってREの学習を行うとき
            predicts_spans = torch.max(spans, dim=1)[1]
            pairs_list_per_batch = PairsModule(predicts_spans).make_pair()

        if Relation_gold_learning_switch:  #NERの結果を使わずにgoldで学習するとき
            pairs_list_per_batch = PairsModule(spans).make_pair()

        data_unit_for_relation = defaultdict(list)
        num_relation = {0:0,1:0,2:0,3:0,4:0,5:0}
        for number_in_minibatch, unique_number in enumerate(n_doc):
            pred_pairs = pairs_list_per_batch[number_in_minibatch]
            gold_pairs = self.REL_LABEL_DICT[unique_number.item()]
            for pred_pair_unit in pred_pairs:
                rel_label_index = self.rel_dic["None"]
                relation_entity_all_label = ()
                for gold_pair_unit in gold_pairs:
                    shaped_gold_unit = [(gold_pair_unit[1][0][0],gold_pair_unit[1][0][1][0]),(gold_pair_unit[1][1][0],gold_pair_unit[1][1][1][0])]
                    if set(pred_pair_unit) == set(shaped_gold_unit):
                        rel_label_index = gold_pair_unit[0]
                        # rel_label_index = 1
                        num_relation[rel_label_index] += 1
                relation_entity_all_label = (pred_pair_unit, rel_label_index)
                data_unit_for_relation[number_in_minibatch].append(relation_entity_all_label)

        for num in range(len(n_doc)):
            data_unit_for_relation.setdefault(num,[])
        # unique_x             = torch.LongTensor([a[0] for a in data_unit_for_relation])
        # rel_word_x           = torch.stack([a[1] for a in data_unit_for_relation], dim=0)
        # rel_pred_x           = torch.LongTensor([a[2] for a in data_unit_for_relation])
        # rel_y                = torch.LongTensor([a[3] for a in data_unit_for_relation]).to(device)
        return data_unit_for_relation




    def pred_span_entity(self, tokens):
        return self.NerModel(tokens)

    def pred_relation(self, tokens, spans, n_doc, down_sampling_switch, Relation_gold_learning_switch):
        # betch_pair = self.make_pair(spans)
        data_unit_for_relation = self.make_pair(spans, n_doc, Relation_gold_learning_switch)

        # if down_sampling_switch:
        #     unique_x, rel_word_x, rel_pred_x, rel_y = self.downsampling(unique_x, rel_word_x, rel_pred_x, rel_y, data_unit_for_relation)

        return self.ReModel(tokens, data_unit_for_relation, 0)



    def forward(self, n_doc, tokens_tensor, attention_mask, NER_RE_switch, down_sampling_switch, y_spans, Relation_gold_learning_switch, is_share_stop):
        ###  SHARE PART ###
        # self.pretrained_BERT_model.eval()
        with torch.no_grad():
            all_encoder_layers = self.bert_model(tokens_tensor, attention_mask=attention_mask)

        if is_share_stop:
            with torch.no_grad():
                rep_word      = all_encoder_layers[self.bert_pooling_layer] # [B, seq_len, 768]
                rep_word = rep_word * attention_mask.view(-1,150,1)
                input_dropout = self.dropout(rep_word)
                h_conv = input_dropout.permute(0,2,1) #[B, 768, seq_len]

        else:
            rep_word      = all_encoder_layers[self.bert_pooling_layer] # [B, seq_len, 768]
            rep_word = rep_word * attention_mask.view(-1,150,1)
            input_dropout = self.dropout(rep_word)
            h_conv = input_dropout.permute(0,2,1) #[B, 768, seq_len]




        ### NER Part ###
        if NER_RE_switch == "NER": #NERの学習のみを行う場合
            logits_spans = self.pred_span_entity(h_conv)
            return logits_spans


        ### RE Part ###
        if NER_RE_switch == "RE": #Relationの学習のみを行う場合
            y_spans = nest_cut(y_spans,self.span_size)
            if Relation_gold_learning_switch: #Relationの学習をGoldで行うかのflag Relationだけの学習を行う=Goldで学習する
                Re_tag, Rel_label = self.pred_relation(h_conv, y_spans, n_doc, down_sampling_switch,Relation_gold_learning_switch)
            else:
                raise ValueError("Relation gold learning switch is OFF")
            return Re_tag, Rel_label



        ### Joint mode ###
        if NER_RE_switch == "Joint": #Joint Learningをするflag = REの学習も行う場合のflag
            logits_spans = self.pred_span_entity(h_conv)

            if Relation_gold_learning_switch: #Relationの学習をGoldで行うかのflag
                Re_tag, Rel_label = self.pred_relation(h_conv, y_spans, n_doc, down_sampling_switch,Relation_gold_learning_switch)
            else:
                Re_tag, Rel_label = self.pred_relation(h_conv, logits_spans, n_doc, down_sampling_switch,Relation_gold_learning_switch)

            return logits_spans, Re_tag, Rel_label

