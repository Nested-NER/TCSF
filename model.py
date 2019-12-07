#!/usr/bin/env python
#-*- coding:utf-8 _*-
"""
@file: model.py
@time: 2019/05/19
# TCSR model
#  class TOI_Pooling defines HIT pooling operation
#  class TOI_CNN_RES defines stacked CNN
#  class TOI_BERT defines the contextual network of DTE using BERT 
"""


import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from utils import clones
from Transformer import Transformer


class TOI_Pooling(nn.Module):
    def __init__(self, input_size, if_gpu, config, pooling_size, pooling_method = 'average'):
        super(TOI_Pooling, self).__init__()
        self.input_size = input_size
        self.config = config
        padding = torch.zeros((input_size, 1))
        self.device = torch.device("cuda:0" if if_gpu else "cpu")
        self.register_buffer('padding', padding)
        self.pooling_size = pooling_size
        self.pooling_method = pooling_method

        if config.if_span_te:
            te_padding = torch.zeros((config.te_size, 1))
            self.register_buffer('te_padding', te_padding)
            self.span_SRN = DTE(config, config.span_te_N , config.span_te_h)
        else:
            config.te_size = 0

    def forward(self, features, tois, embeddings):
        result = []
        res_span_te=[]
        for i in range(len(tois)):
            result.append(self.HAT_TOI_Pooling(features[i], tois[i]))
            if self.config.if_span_te:
                span_te, span_mean = self.span_mean_te(embeddings[i],tois[i])
                res_span_te.append(span_te)   # result: sentence list of  span_num * dims

        if self.config.if_span_te:
            return torch.cat([torch.cat(result, 0), torch.cat(res_span_te,0)],-1), np.cumsum([len(s) for s in tois]).astype(np.int32)
        else:
            return torch.cat(result, 0), np.cumsum([len(s) for s in tois]).astype(np.int32)

    def span_mean_te(self, feature, tois):

        start = tois[:, 0]
        end = tois[:, 1]
        tois_len = Variable(torch.FloatTensor(end - start), requires_grad=False).cuda()
        ret_cs = torch.cat([Variable(self.te_padding, requires_grad=False), torch.cumsum(feature, 1)], dim=1)
        spans_mean = (ret_cs[:, end] - ret_cs[:, start]) / tois_len

        mask = torch.ones(1, len(tois), len(tois)).cuda()  # .cuda()
        input = spans_mean.transpose(0, 1).unsqueeze(0)
        output = self.span_SRN(mask, input, None, None, None, None)
        return output.squeeze(0), spans_mean

    def HAT_TOI_Pooling(self, feature, tois):
        if feature.dim() != 2:
            feature = feature.unsqueeze(1)

        if self.pooling_method == 'average':
            start = tois[:, 0]
            end = tois[:, 1]
            return self.hit(feature, start, end).t()

    def hit(self, feature, start, end):
        # the avg length lass than 3, to speed training process base the short sample, calculate in site
        ret = torch.zeros(self.pooling_size + 2, feature.size(0), len(start)).cuda()
        ret[0, :, :] = feature[:, start]
        length = end - start
        length_greater_1 = np.where(length > 1)
        ret[self.pooling_size + 1, : , length_greater_1] = feature[: , end[length_greater_1] - 1].unsqueeze(1)
        for i in range(1, self.pooling_size + 1):
            length_fixed = np.where(length == i + 2)
            if len(length_fixed[0]) == 0: continue
            tmp = []
            for j in range(1, i+1):
                tmp.append(feature[:, start[length_fixed]+j])
            tmp = torch.cat(tmp, 0).unsqueeze(1)
            tmp = tmp.reshape(-1, feature.size(0), 1, tmp.size(2))
            ret[1:i+1, : ,length_fixed] = tmp
        length_max = np.where(length > self.pooling_size + 2)
        if len(length_max[0]) == 0:
            return ret.reshape(-1, ret.size(2))
        length_multi = (length[length_max] - 2) // self.pooling_size
        length_remain = (length[length_max] - 2) % self.pooling_size
        sum_feature = torch.cat([Variable(self.padding, requires_grad=False), torch.cumsum(feature, 1)], dim=1)
        for i in range(1, self.pooling_size + 1):
            ret[i, : , length_max] = ((sum_feature[:, start[length_max] + 1 + i * length_multi + length_remain] - sum_feature[:, start[length_max] + 1 + (i-1) * length_multi]) \
                                     / torch.FloatTensor(length_multi + length_remain).cuda()).unsqueeze(1)
        return ret.reshape(-1, ret.size(2))

class TOI_CNN_RES(nn.Module):
    def __init__(self, input_dim, output_dim, kernal_size = 3, dropout_rate = 0.5):
        super(TOI_CNN_RES, self).__init__()
        self.resnet = nn.Sequential(
            nn.Conv1d(input_dim, input_dim,
                      kernel_size = kernal_size, padding = (kernal_size - 1) // 2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )
        self.short_cut = nn.Sequential(
            nn.Conv1d(input_dim, output_dim,
                      kernel_size=1)
        )
        self.ret = nn.ReLU()

    def forward(self, input):
        return self.ret(self.resnet(input) + self.short_cut(input))

class TOI_BERT(nn.Module):
    def __init__(self, config):
        super(TOI_BERT, self).__init__()
        self.config = config
        self.outfile = None

        self.input_size_bert = config.input_size_bert
        self.input_size = self.input_size_bert if config.use_bert else 0

        if self.config.if_DTE:
            self.dte = DTE(config, self.config.N, self.config.h)
            self.input_size += self.dte.input_size

        if self.config.use_cnn:
            self.res_nets = clones(TOI_CNN_RES(self.input_size, self.input_size, kernal_size=self.config.kernel_size), self.config.cnn_block)
        else:
            self.project = nn.Sequential(
               nn.Linear(self.input_size, self.input_size),
               nn.Dropout(0.5),
               nn.ReLU()
           )



        self.hat_1 = TOI_Pooling(self.input_size, self.config.if_gpu, self.config, self.config.hit_pooling_size)

        self.pooling_size = 2 + self.config.hit_pooling_size

        self.one_step_to_share=nn.Sequential(
            nn.Linear(self.input_size * self.pooling_size + config.te_size, self.config.nested_depth_fc_size),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        self.one_step_to_heaven = nn.Sequential(
            nn.Linear(self.config.nested_depth_fc_size, self.config.label_kinds),
        )

        self.one_step_to_hell = nn.Sequential(
            nn.Linear(self.config.nested_depth_fc_size, self.config.nested_depth),
        )

        if self.config.fusion:
            self.fusion = nn.Sequential(
                nn.Softmax(dim = 0)
            )
            self.fusion_parameters = torch.nn.Parameter(torch.ones(config.fusion_layer, 1))
            self.fusion_gamma = torch.nn.Parameter(torch.ones(1))

        self.cls_ce_loss = nn.CrossEntropyLoss()

    def forward(self, mask_batch, word_batch, char_batch, char_len_batch, pos_batch, toi_batch, bert_hidden, entity_idx):
        softmax_wight = None
        if self.config.use_bert:
            if not self.config.fusion_sum:
                bert_hidden_out = torch.zeros(bert_hidden.size()).cuda() if self.config.if_gpu else torch.zeros(bert_hidden.size())
            else:
                softmax_wight = self.fusion(self.fusion_parameters)
                bert_hidden_out = torch.zeros(bert_hidden.size(0),bert_hidden.size(2),bert_hidden.size(3)).cuda() if self.config.if_gpu \
                    else torch.zeros(bert_hidden.size(0),bert_hidden.size(2),bert_hidden.size(3))

            if self.config.fusion_sum:
                if self.config.use_last_four:
                    #embedding
                    for each in range(bert_hidden.size(1) - 4 - 1, bert_hidden.size(1) - 1):
                        bert_hidden_out += bert_hidden[:, each, :]

                    bert_hidden_out /= 4
                else:
                    for each in range(bert_hidden.size(0)):
                        bert_hidden_out[each] = (bert_hidden[each].reshape(bert_hidden[each].size(0), -1) \
                            * softmax_wight).reshape(bert_hidden[each].size()).sum(0)
                    bert_hidden_out *= 0.3
            else:
                bert_hidden_out = bert_hidden



        if self.config.if_DTE:
            dte_features = self.dte(mask_batch, word_batch, char_batch, char_len_batch, pos_batch)
            if self.config.use_bert:
                bert_hidden_out = torch.cat((bert_hidden_out, dte_features), -1)
            else:
                bert_hidden_out = dte_features

        if self.config.use_cnn:
            features = bert_hidden_out.transpose(1,2)
            for res_net in self.res_nets:
                features = res_net(features)
            features = features.transpose(1, 2)

        else:
            features = self.project(bert_hidden_out)

        features, toi_section = self.hat_1(features.transpose(1, 2), toi_batch,  dte_features.transpose(1, 2))

        features=self.one_step_to_share(features)

        features_label = self.one_step_to_heaven(features)

        if len(entity_idx):
            features = features[entity_idx, : ]
            features_nested_depth = self.one_step_to_hell(features)
        else:
            features_nested_depth = []

        return features_label, features_nested_depth, toi_section

    def load_vector(self):
        self.dte.load_vector()

    def calc_loss(self, cls_s, gold_label):
        return self.cls_ce_loss(cls_s, gold_label)


class CharEmbed(nn.Module):
    def __init__(self, char_kinds, embedding_size):
        super(CharEmbed, self).__init__()
        self.char_embed = nn.Embedding(char_kinds, embedding_size)
        self.char_bilstm = nn.LSTM(embedding_size, embedding_size, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, char_id, char_len):
        char_vec = self.char_embed(char_id)
        char_vec = torch.cat(tuple(char_vec))
        chars_len = np.concatenate(char_len)
        perm_idx = chars_len.argsort(0)[::-1].astype(np.int32)
        back_idx = perm_idx.argsort().astype(np.int32)
        pack = pack_padded_sequence(char_vec[perm_idx], chars_len[perm_idx], batch_first=True)
        lstm_out, (hid_states, cell_states) = self.char_bilstm(pack)
        return torch.cat(tuple(hid_states), 1)[back_idx]


class DTE(nn.Module):
    def __init__(self, config, NN, hh):
        super(DTE, self).__init__()
        self.config = config
        self.input_size = config.word_embedding_size
        if self.config.if_pos:
            self.pos_embed = nn.Embedding(config.pos_tag_kinds, config.pos_embedding_size)
            self.input_size += config.pos_embedding_size
        if self.config.if_char:
            self.char_embed = CharEmbed(config.char_kinds, config.char_embedding_size)
            self.input_size += (2 * config.char_embedding_size)
        self.word_embed = nn.Embedding(config.word_kinds, config.word_embedding_size)
        self.input_dropout = nn.Dropout(config.dropout)

        if self.config.if_transformer:
            self.transformer = Transformer(d_model=self.input_size, N=NN, h=hh,
                                           dropout=0.1, bidirectional=self.config.if_bidirectional)

    def forward(self, mask_batch, word_batch, char_batch, char_len_batch, pos_batch, loc_span=None):
        if char_batch is None:
            word_vec = word_batch
        else:
            word_vec = self.word_embed(word_batch)
            if self.config.if_char:
                char_vec = self.char_embed(char_batch, char_len_batch)
                word_vec = torch.cat([word_vec, char_vec.view(word_vec.shape[0], word_vec.shape[1], -1)], 2)
            if self.config.if_pos:
                pos_vec = self.pos_embed(pos_batch)
                word_vec = torch.cat([word_vec, pos_vec], 2)
            word_vec = self.input_dropout(word_vec)

        if self.config.if_transformer:
            word_vec = self.transformer(word_vec, mask_batch)

        return word_vec

    def load_vector(self):
        with open(self.config.get_pkl_path("word2vec"), "rb") as f:
            vectors = pickle.load(f)
            w_v = torch.Tensor(vectors)
            self.word_embed.weight = nn.Parameter(w_v)
            if self.config.if_freeze:
                self.word_embed.weight.requires_grad = False
            print(f"Loading from {self.config.get_pkl_path('word2vec')}")
