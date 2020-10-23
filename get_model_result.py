# -*- coding: utf-8 -*-
__author__ = ''
__data__ = '2019/7/23'

from utils import select_toi, generate_mask, dfs
import torch
from torch.autograd import Variable
import numpy as np


class ModelInRuntime:
    def __init__(self, args):
        model, bertModel, tokenizer, config , len = args
        self.model = model
        self.bertModel = bertModel
        self.config = config
        self.tokenizer = tokenizer
        self.word_dic_in = {}
        self.word_dic_out = {}
        self.cnt = 0
        self.epoch_cnt = 0
        self.text_to_bert_out = [None] * len * self.config.batch_size
        self.text_to_bert = {}

    @classmethod
    def instance(cls, args):
        if hasattr(cls, "_instance"):
            return cls._instance
        cls._instance = cls(args)
        return cls._instance

    def setRuntimeData(self, runningData):
        self.setData(runningData)
        num_sum = 0
        for each in self.toi_batch:
            num_sum += len(each)
        self.entity_idx_in_toi_box = [each for each in range(num_sum)]
        self.preProcessData()

    def setTrainData(self, runningData):
        self.setData(runningData)
        self.entity_idx_in_toi_box  = []
        count_toi_box_numbers = 0
        for idx in range(len(self.entity_batch)):
            for each in self.entity_batch[idx]:
                id = self.toi_batch[idx].index(each)
                self.entity_idx_in_toi_box.append(id + count_toi_box_numbers)
            count_toi_box_numbers += len(self.toi_batch[idx])
        tmp = []
        for each in self.entity_batch:
            if len(each):
                tmp.extend(each)
        all_toi = np.concatenate(self.toi_batch)
        if len(tmp):
            for each in self.entity_idx_in_toi_box:
                assert tuple(all_toi[each]) in tmp

        self.preProcessData()

    def setTestData(self, runningData):
        self.setData(runningData)
        self.entity_idx_in_toi_box = []
        id = 0
        for item in self.toi_batch:
            for it in item:
                self.entity_idx_in_toi_box.append(id)
                id += 1
        self.preProcessData()

    def setData(self, runningData):
        self.processDataAll = runningData
        # get data_item
        self.word_batch, \
        self.char_batch, \
        self.char_len_batch, \
        self.pos_tag_batch, \
        self.entity_batch, \
        self.toi_batch, \
        self.word_origin_batch = self.processDataAll

    def preProcessData(self):

        self.toi_box_batch, self.label_batch, _ = select_toi(self.toi_batch)

        self.toi_box_entity, _, self.space_list_entity = select_toi(self.entity_batch)
        self.word_batch_var = self.whetherUseGpu(Variable(torch.LongTensor(np.array(self.word_batch))),
                                                 self.config.if_gpu)
        self.mask_batch_var = self.whetherUseGpu(generate_mask(self.word_batch_var.shape), self.config.if_gpu)
        self.char_batch_var = self.whetherUseGpu(Variable(torch.LongTensor(np.array(self.char_batch))),
                                                 self.config.if_gpu)
        self.pos_tag_batch_var = self.whetherUseGpu(Variable(torch.LongTensor(np.array(self.pos_tag_batch))),
                                                    self.config.if_gpu)
        self.gold_label_vec = self.whetherUseGpu(Variable(torch.LongTensor(np.hstack(self.label_batch))),
                                                 self.config.if_gpu)
        self.entity_nested_depth = []
        for each_entiy_list in self.entity_batch:
            gt_layer = [-1] * len(each_entiy_list)
            for id in range(len(gt_layer)):
                if gt_layer[id] == -1:
                    dfs(id, each_entiy_list, gt_layer)
            self.entity_nested_depth.append(gt_layer)
        self.entity_nested_depth = np.hstack(self.entity_nested_depth)
        self.entity_nested_depth[
            np.where(self.entity_nested_depth >= self.config.nested_depth)] = self.config.nested_depth - 1
        self.entity_nested_depth = self.whetherUseGpu(Variable(torch.LongTensor(self.entity_nested_depth)),
                                                      self.config.if_gpu)


    def whetherUseGpu(self, value, flag):
        return value.cuda() if flag else value

    def runClassification(self):
        label, nested_depth, section = self.model(self.mask_batch_var, self.word_batch_var, self.char_batch_var, self.char_len_batch,
                               self.pos_tag_batch_var, self.toi_box_batch, self.hiddenList, self.entity_idx_in_toi_box)
        self.nested_depth = nested_depth
        return label, section, self.gold_label_vec

    def getNestedData(self):
        if len(self.toi_box_entity):
            return True
        else:
            return False

    def runNestedDepth(self):
        return self.nested_depth, None, self.entity_nested_depth
