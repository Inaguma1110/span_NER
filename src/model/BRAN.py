import itertools
import random
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
    def __init__(self, model_hparams_dic, span_size):
        super(SpanHead, self).__init__()
        self.embedding_dim  = model_hparams_dic['embedding_dim']
        self.window_size    = model_hparams_dic['window_size']
        self.hidden_dim     = model_hparams_dic['hidden_dim']
        self.SpanHeadFctype = model_hparams_dic['spanheadfctype']
        self.span_size = span_size

        self.span_filter   = nn.Conv1d(self.embedding_dim, self.hidden_dim, self.span_size)
        self.conv = nn.Conv1d(self.hidden_dim, self.hidden_dim, self.window_size, padding=((self.window_size//2), ))
        self.linear        = nn.Linear(self.hidden_dim, 2)

    def forward(self, inputs, SpanHeadFctype):
        paded       = F.pad(inputs, (0,self.span_size-1))        # [B, 768, seq_len + span_size]
        spaned      = F.relu(self.span_filter(paded))            # [B, D, seq_len]
        spaned      = F.relu(self.conv(spaned))
        if self.SpanHeadFctype == 'share':
            return spaned

        if self.SpanHeadFctype == 'each':
            lineard     = F.relu(self.linear(spaned.permute(0,2,1))) # [B, D, 2]
            logits_span = lineard.permute(0,2,1)                     # [B, seq_len, 2]
            return logits_span

class NamedEntityExtractionModule(nn.Module):
    def __init__(self, model_hparams_dic, span_size):
        super(NamedEntityExtractionModule, self).__init__()
        self.span_size = span_size
        self.hidden_dim     = model_hparams_dic['hidden_dim']
        self.SpanHeadFctype = model_hparams_dic['spanheadfctype']
        self.heads = nn.ModuleList([SpanHead(model_hparams_dic, span_size) for span_size in range(1, span_size+1)])
        self.share_linear = nn.Linear(self.hidden_dim, 2)
    def forward(self, tokens):
        if self.SpanHeadFctype == 'share':
            spans = [self.heads[s_x](tokens,self.SpanHeadFctype) for s_x in range(self.span_size)]
            spans = torch.stack(spans, dim = 0)
            lineared = self.share_linear(spans.permute(0,1,3,2))
            logits_spans = lineared.permute(0,1,3,2)
            return logits_spans

        if self.SpanHeadFctype == 'each':
            logits_spans = [self.heads[s_x](tokens, self.SpanHeadFctype) for s_x in range(self.span_size)]
            logits_spans = torch.stack(logits_spans, dim = 0)
            return logits_spans

class SimpleAttention(nn.Module):
    def __init__(self, model_hparams_dic):
        super(SimpleAttention, self).__init__()
        self.embedding_dim  = model_hparams_dic['embedding_dim']
        self.window_size    = model_hparams_dic['window_size']
        self.hidden_dim     = model_hparams_dic['hidden_dim']
        self.num_MHA        = model_hparams_dic['num_MHA']

        self.q_dense_layer = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.k_dende_layer = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.v_dense_layer = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.Multihead    = nn.MultiheadAttention(embed_dim = self.embedding_dim, num_heads = self.num_MHA)

    def forward(self, h_conv, padding_mask):
        query = self.q_dense_layer(h_conv.permute(0,2,1)).permute(1,0,2)
        key   = self.k_dende_layer(h_conv.permute(0,2,1)).permute(1,0,2)
        val   = self.v_dense_layer(h_conv.permute(0,2,1)).permute(1,0,2)
        trigrep, trig_attn = self.Multihead(query, key, val, key_padding_mask = ~padding_mask.bool()) # No trig
        return trigrep, trig_attn


class TriggerCatAttention(nn.Module):
    def __init__(self, model_hparams_dic):
        super(TriggerCatAttention, self).__init__()
        self.embedding_dim  = model_hparams_dic['embedding_dim']
        self.window_size    = model_hparams_dic['window_size']
        self.hidden_dim     = model_hparams_dic['hidden_dim']
        self.num_MHA        = model_hparams_dic['num_MHA']

        self.trig_dim = self.embedding_dim

        self.trig_embedding_layer = nn.Embedding(2, self.trig_dim)
        self.q_dense_layer = nn.Linear(self.embedding_dim*2, self.embedding_dim)
        self.k_dende_layer = nn.Linear(self.embedding_dim,   self.embedding_dim)
        self.v_dense_layer = nn.Linear(self.embedding_dim,   self.embedding_dim)
        self.Multihead    = nn.MultiheadAttention(embed_dim = self.embedding_dim, num_heads = self.num_MHA)

    def forward(self, h_conv, trigger_vecs, padding_mask):
        trigger_vecs = self.trig_embedding_layer(trigger_vecs)
        h_conv_trig = torch.cat((h_conv.permute(0,2,1),trigger_vecs),dim=2)
        query = self.q_dense_layer(h_conv_trig).permute(1,0,2)
        key   = self.k_dende_layer(h_conv.permute(0,2,1)).permute(1,0,2)
        val   = self.v_dense_layer(h_conv.permute(0,2,1)).permute(1,0,2)
        trigrep, trig_attn = self.Multihead(query, key, val, key_padding_mask = ~padding_mask.bool())

        return trigrep, trig_attn


class ResidualKeyValueAttention(nn.Module):
    def __init__(self, model_hparams_dic):
        super(ResidualKeyValueAttention, self).__init__()
        self.embedding_dim  = model_hparams_dic['embedding_dim']
        self.window_size    = model_hparams_dic['window_size']
        self.hidden_dim     = model_hparams_dic['hidden_dim']
        self.num_MHA        = model_hparams_dic['num_MHA']

        self.q_dense_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.Multihead     = nn.MultiheadAttention(embed_dim = self.embedding_dim, num_heads = self.num_MHA)

    def forward(self, h_conv, trigger_vecs, padding_mask):
        query = self.q_dense_layer(h_conv.permute(0,2,1)).permute(1,0,2)
        trigger_vecss = torch.cat([trigger_vecs.unsqueeze(1)]*h_conv.size(1),dim=1)
        trig_rep = (h_conv * trigger_vecss) + h_conv        
        trig_rep = trig_rep.permute(2,0,1)
        trigrep, trig_attn = self.Multihead(query, trig_rep, trig_rep, key_padding_mask = ~padding_mask.bool())
        return trigrep, trig_attn



class KeyValueAttention(nn.Module):
    def __init__(self, model_hparams_dic):
        super(KeyValueAttention, self).__init__()
        self.embedding_dim  = model_hparams_dic['embedding_dim']
        self.window_size    = model_hparams_dic['window_size']
        self.hidden_dim     = model_hparams_dic['hidden_dim']
        self.num_MHA        = model_hparams_dic['num_MHA']

        self.q_dense_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.Multihead     = nn.MultiheadAttention(embed_dim = self.embedding_dim, num_heads = self.num_MHA)

    def forward(self, h_conv, trigger_vecs, padding_mask):
        query = self.q_dense_layer(h_conv.permute(0,2,1)).permute(1,0,2)
        trigger_vecss = torch.cat([trigger_vecs.unsqueeze(1)]*h_conv.size(1),dim=1)
        trig_rep = (h_conv * trigger_vecss)
        trig_rep = trig_rep.permute(2,0,1)
        trigrep, trig_attn = self.Multihead(query, trig_rep, trig_rep, key_padding_mask = ~padding_mask.bool())
        return trigrep, trig_attn

class RelationExtractionModule(nn.Module):
    def __init__(self, vocab, model_hparams_dic, rel_dic):
        super(RelationExtractionModule, self).__init__()
        self.rel_size = len(rel_dic)
        # self.rel_size = 2

        self.embedding_dim    = model_hparams_dic['embedding_dim']
        self.window_size      = model_hparams_dic['window_size']
        self.hidden_dim       = model_hparams_dic['hidden_dim']

        self.rel_conv         = nn.Conv1d(self.embedding_dim, self.embedding_dim, self.window_size, padding=((self.window_size//2), ))
        self.rel_conv2        = nn.Conv1d(self.embedding_dim, self.embedding_dim, self.window_size, padding=((self.window_size//2), ))

        # self.rel_linear      = nn.Linear(self.hidden_dim, self.rel_size)
        self.rel_linear       = nn.Linear(self.hidden_dim, self.rel_size)
        self.linear1          = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.linear2          = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear0          = nn.Linear(self.embedding_dim,self.embedding_dim)

        self.Pooling_1        = nn.MaxPool1d(4, stride=2)
        self.Pooling_2        = nn.MaxPool1d(4, stride=2)

    def forward(self, rel_word_x, data_unit_for_relation, trigger_vecss, trig_attn):
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
        
        rel_logits = torch.stack(tuple(rel_logits),dim=0)
        rel_labels = torch.stack(tuple(rel_labels),dim=0)
        return rel_logits, trig_attn, rel_labels


class MyModel(nn.Module):
    def __init__(self, span_size, config, vocab, model_hparams_dic, rel_dic, REL_LABEL_DICT,doc_correspnd_info_dict):
        super(MyModel, self).__init__()
        self.bert_model = JapaneseBertPretrainedModel(config, vocab)
        self.span_size = span_size
        self.rel_dic = rel_dic
        self.REL_LABEL_DICT = REL_LABEL_DICT
        self.doc_correspnd_info_dict = doc_correspnd_info_dict
        self.bert_pooling_layer = 0
        self.embedding_dim = model_hparams_dic['embedding_dim']
        self.window_size = model_hparams_dic['window_size']
        self.headtype    = model_hparams_dic['h_2head']
        self.dropout     = nn.Dropout(p=0.5)
        self.conv = nn.Conv1d(self.embedding_dim, self.embedding_dim, self.window_size, padding=((self.window_size//2), ))

        # h_2 Vector Process 
        self.SimpleAttentionLayer                = SimpleAttention(model_hparams_dic)
        self.TrigCatLayer                        = TriggerCatAttention(model_hparams_dic)
        self.KeyValTriggerAttentionLayer         = KeyValueAttention(model_hparams_dic)
        self.KeyValTriggerAttentionResidualLayer = ResidualKeyValueAttention(model_hparams_dic)
        self.NerModel = NamedEntityExtractionModule(model_hparams_dic, span_size)
        self.ReModel = RelationExtractionModule(vocab, model_hparams_dic, rel_dic)
    
    def decision_h_2_head(self, h_conv, trigger_vecs, padding_mask, headtype):
        if headtype == 'SimpleAttention':
            trigrep, trig_attn = self.SimpleAttentionLayer(h_conv, padding_mask) #trigrep = [Seq_len, B, 768]
            return trigrep, trig_attn
        if headtype == 'TrigCatQuery':
            trigrep, trig_attn = self.TrigCatLayer(h_conv, trigger_vecs, padding_mask)
            return trigrep, trig_attn
        if headtype == 'TrigKeyVal':
            trigrep, trig_attn = self.KeyValTriggerAttentionLayer(h_conv, trigger_vecs, padding_mask)
            return trigrep, trig_attn
        if headtype == 'TrigKeyValResidual':
            trigrep, trig_attn = self.KeyValTriggerAttentionResidualLayer(h_conv, trigger_vecs, padding_mask)
            return trigrep, trig_attn
        else:
            print('error')

    def make_pair(self, spans, n_doc, Relation_gold_learning_switch):
        n_doc = n_doc.to('cpu').detach().numpy().copy().squeeze()

        if not Relation_gold_learning_switch: # NERの結果を使ってREの学習を行うとき
            predicts_spans = torch.max(spans, dim=2)[1].permute(1,0,2)
            pairs_list_per_batch = PairsModule(predicts_spans).make_pair()

        if Relation_gold_learning_switch:  #NERの結果を使わずにgoldで学習するとき
            pairs_list_per_batch = PairsModule(spans).make_pair()

        data_unit_for_relation = defaultdict(list)
        num_relation = {0:0,1:0,2:0,3:0,4:0,5:0}
        # pdb.set_trace()
        for number_in_minibatch, unique_number in enumerate(n_doc):
            try:    
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
            except KeyError:
                pass
        for num in range(len(n_doc)):
            data_unit_for_relation.setdefault(num,[])

        cc = 0
        for k, v in data_unit_for_relation.items():
            cc += len(v)
        if cc == 0 :
            print('error make relation data')
            pdb.set_trace()
        return data_unit_for_relation

    def pred_span_entity(self, tokens):
        return self.NerModel(tokens)

    def pred_relation(self, tokens, spans, n_doc, trigger_vecss, trig_attn, Relation_gold_learning_switch):
        data_unit_for_relation = self.make_pair(spans, n_doc, Relation_gold_learning_switch)
        return self.ReModel(tokens, data_unit_for_relation, trigger_vecss, trig_attn)

    def forward(self, n_doc, tokens_tensor, trigger_vecs, padding_mask, NER_RE_switch, y_spans, Relation_gold_learning_switch, is_share_stop):
        ###  SHARE PART ###
        # self.pretrained_BERT_model.eval()
        with torch.no_grad():
            all_encoder_layers = self.bert_model(tokens_tensor, attention_mask=padding_mask)

        ### NER Part ###
        if NER_RE_switch == "NER": #NERの学習のみを行う場合
            if is_share_stop:
                with torch.no_grad():
                    rep_word      = all_encoder_layers[self.bert_pooling_layer] # [B, seq_len, 768]
                    rep_word = rep_word * padding_mask.view(-1,150,1)
                    input_dropout = self.dropout(rep_word)
                    h_conv = input_dropout.permute(0,2,1) #[B, 768, seq_len]
                    h_conv = self.conv(h_conv)
            else:
                rep_word      = all_encoder_layers[self.bert_pooling_layer] # [B, seq_len, 768]
                rep_word = rep_word * padding_mask.view(-1,150,1)
                input_dropout = self.dropout(rep_word)
                h_conv = input_dropout.permute(0,2,1) #[B, 768, seq_len]
            logits_spans = self.pred_span_entity(h_conv)
            return logits_spans

        ### RE Part ###
        if NER_RE_switch == "RE": #Relationの学習のみを行う場合
            if is_share_stop:
                with torch.no_grad():
                    rep_word      = all_encoder_layers[self.bert_pooling_layer] # [B, seq_len, 768]
                    rep_word = rep_word * padding_mask.view(-1,150,1)
                    input_dropout = self.dropout(rep_word)
                    h_conv = input_dropout.permute(0,2,1) #[B, 768, seq_len]

            else:
                rep_word      = all_encoder_layers[self.bert_pooling_layer] # [B, seq_len, 768]
                rep_word = rep_word * padding_mask.view(-1,150,1)
                input_dropout = self.dropout(rep_word)
                h_conv = input_dropout.permute(0,2,1) #[B, 768, seq_len]

            trigrep, trig_attn = self.decision_h_2_head(h_conv, trigger_vecs, padding_mask, self.headtype)
            hidden = trigrep.permute(1,2,0)
            y_spans = nest_cut(y_spans,self.span_size)
            if Relation_gold_learning_switch: #Relationの学習をGoldで行うかのflag Relationだけの学習を行う=Goldで学習する
                re_tag, trig_attn, re_label = self.pred_relation(hidden, y_spans, n_doc, trigger_vecs, trig_attn, Relation_gold_learning_switch)
            else:
                raise ValueError("Relation gold learning switch is OFF")
            return re_tag, trig_attn, re_label

        ### Joint mode ###
        if NER_RE_switch == "Joint": #Joint Learningをするflag = REの学習も行う場合のflag
            if is_share_stop:
                with torch.no_grad():
                    rep_word      = all_encoder_layers[self.bert_pooling_layer] # [B, seq_len, 768]
                    rep_word = rep_word * padding_mask.view(-1,150,1)
                    input_dropout = self.dropout(rep_word)
                    h_conv = input_dropout.permute(0,2,1) #[B, 768, seq_len]
            else:
                rep_word = all_encoder_layers[self.bert_pooling_layer] # [B, seq_len, 768]
                rep_word = rep_word * padding_mask.view(-1,150,1)
                input_dropout = self.dropout(rep_word)
                h_conv = input_dropout.permute(0,2,1) #[B, 768, seq_len]
            trigrep, trig_attn = self.decision_h_2_head(h_conv, trigger_vecs, padding_mask, self.headtype)
            hidden = trigrep.permute(1,2,0)

            logits_spans = self.pred_span_entity(h_conv)
            if Relation_gold_learning_switch: #Relationの学習をGoldで行うかのflag
                re_tag, trig_attn, re_label = self.pred_relation(hidden, y_spans, n_doc, trigger_vecs, trig_attn, Relation_gold_learning_switch)
            else:
                re_tag, trig_attn, re_label = self.pred_relation(hidden, logits_spans, n_doc, trigger_vecs, trig_attn, Relation_gold_learning_switch)
            return logits_spans, re_tag, trig_attn, re_label


        if NER_RE_switch == 'Joint_ner_share_stop':
            if is_share_stop:
                with torch.no_grad():
                    rep_word      = all_encoder_layers[self.bert_pooling_layer] # [B, seq_len, 768]
                    rep_word = rep_word * padding_mask.view(-1,150,1)
                    input_dropout = self.dropout(rep_word)
                    h_conv = input_dropout.permute(0,2,1) #[B, 768, seq_len]
                    logits_spans = self.pred_span_entity(h_conv)

                rep_word = all_encoder_layers[self.bert_pooling_layer] # [B, seq_len, 768]
                rep_word = rep_word * padding_mask.view(-1,150,1)
                input_dropout = self.dropout(rep_word)
                h_conv = input_dropout.permute(0,2,1) #[B, 768, seq_len]
                trigrep, trig_attn = self.decision_h_2_head(h_conv, trigger_vecs, padding_mask, self.headtype)
                hidden = trigrep.permute(1,2,0)
                if Relation_gold_learning_switch: #Relationの学習をGoldで行うかのflag
                    re_tag, trig_attn, re_label = self.pred_relation(hidden, y_spans, n_doc, trigger_vecs, trig_attn, Relation_gold_learning_switch)
                else:
                    re_tag, trig_attn, re_label = self.pred_relation(hidden, logits_spans, n_doc, trigger_vecs, trig_attn, Relation_gold_learning_switch)
                return logits_spans, re_tag, trig_attn, re_label


            else:
                rep_word = all_encoder_layers[self.bert_pooling_layer] # [B, seq_len, 768]
                rep_word = rep_word * padding_mask.view(-1,150,1)
                input_dropout = self.dropout(rep_word)
                h_conv = input_dropout.permute(0,2,1) #[B, 768, seq_len]            
                trigrep, trig_attn = self.decision_h_2_head(h_conv, trigger_vecs, padding_mask, self.headtype)
                hidden = trigrep.permute(1,2,0)
                logits_spans = self.pred_span_entity(h_conv)

                if Relation_gold_learning_switch: #Relationの学習をGoldで行うかのflag
                    re_tag, trig_attn, re_label = self.pred_relation(hidden, y_spans, n_doc, trigger_vecs, trig_attn, Relation_gold_learning_switch)
                else:
                    re_tag, trig_attn, re_label = self.pred_relation(hidden, logits_spans, n_doc, trigger_vecs, trig_attn, Relation_gold_learning_switch)
                return logits_spans, re_tag, trig_attn, re_label