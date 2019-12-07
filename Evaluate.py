import numpy as np
import torch.nn.functional as F
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer
from config import config
from tqdm import tqdm
from utils import check_contain, batch_split
from get_model_result import ModelInRuntime

tokenizer = BertTokenizer.from_pretrained(f"bert-{config.bert_config}-uncased")

class Evaluate:
    def __init__(self, model, config):
        self.model = model    # type: TOI_BERT
        self.config = config
        self.outfile = open(config.get_result_path(), "w") if config.if_output else None
        self.classes_num = len(self.model.config.id2label) - 1
        self.cnt_overlapped = 0

    def get_f1(self, batch_zip, bert_model):
        self.init()
        batch_zip = list(batch_zip)
        self.config.mode = "Test"
        bert_model.eval()
        for each_batch in tqdm(batch_zip):
            self.model.eval()
            runtimeModel = ModelInRuntime.instance((self.model, bert_model, tokenizer, self.config,  len(list(batch_zip))))
            runtimeModel.setRuntimeData(each_batch)
            cls_s, toi_section, aim = runtimeModel.runClassification()
            cls_s_label = batch_split(cls_s, toi_section)
            cls, _, _  = runtimeModel.runNestedDepth()

            for i in range(len(runtimeModel.word_batch)):
                pred_entities = self.predict(runtimeModel.toi_box_batch[i], cls_s_label[i], runtimeModel.entity_batch[i], runtimeModel.word_batch_var[i])
                pred_entities_nested_label = []
                self.evaluate_detail(runtimeModel.entity_batch[i], pred_entities, pred_entities_nested_label)

        return self.calc_f1()

    def filter(self, label, depth):
        deteleList = []
        for id in range(len(label)):
            if(label[id][2] == 0):
                deteleList.append(id)

        label = np.delete(label, deteleList, 0)
        depth = np.delete(depth, deteleList, 0)
        return label.tolist(), depth.tolist()


    def calc_f1(self):
        print("------------------------------------------")
        df, cls_f1 = self.get_count_dataframe(self.confusion_matrix, list(self.model.config.id2label)[1:])
        print(df)

        return cls_f1

    def predict(self, tois, cls_s, entities, word_ids):
        words = [self.model.config.id2word[word] for word in word_ids]
        self.write_result(None, None, None, entities, words, 'begin')
        if cls_s.shape[0] == 0:
            self.write_result(None, None, None, None, None, 'end')
            return []

        cls_score = F.softmax(cls_s, dim=1).data.cpu().numpy()
        cls = np.argmax(cls_score, axis=1)
        cls_score = np.max(cls_score, axis=1)

        pre_index = np.where(cls != 0)[0]
        tois_ = tois[pre_index]
        cls = cls[pre_index]
        cls_score = cls_score[pre_index]
        self.renewal_F_result(tois_, cls, entities, self.toi_F)
        self.write_result(tois_, cls, cls_score, entities, words, 'TOI')

        self.renewal_F_result(tois_, cls, entities, self.score_F)
        self.write_result(tois_, cls, cls_score, entities, words, 'score filter')


        return [(tois_[i][0], tois_[i][1], cls[i]) for i in range(len(tois_))]

    def write_result(self, tois, cls, cls_score, gold_entities, words, type):
        if not self.config.if_output:
            return

        if type == 'begin':
            self.outfile.write(' '.join(words) + '\ngt: ')
            for gt in gold_entities:
                self.outfile.write(self.model.config.id2label[gt[2]] + ' [' + ' '.join(words[gt[0]: gt[1]]) + ']  ')
            self.outfile.write('\n')
            return

        if type == 'end':
            self.outfile.write('--------------------------------------------------------\n\n')
            return

        pre_is_right = np.zeros(len(tois))
        for i in range(len(tois)):
            pre = (tois[i][0], tois[i][1], cls[i])
            if pre in gold_entities:
                pre_is_right[i] = 1
        pre_is_right = pre_is_right.astype(np.int32)
        r_e = ["F", "T"]
        self.outfile.write('--------------------------------------------------------\n' + type + '\n')
        for i in range(len(tois)):
            self.outfile.write('\t'.join(
                [r_e[pre_is_right[i]], self.model.config.id2label[cls[i]], str(tois[i]),
                 str(np.round(cls_score[i], 3)), '[' + ' '.join(words[tois[i][0]: tois[i][1]]) + ']']) + '\n')
        self.outfile.write('\n')

    def renewal_F_result(self, tois, pred_cls, gold_entities, dic):
        pred_entities = [(tois[i][0], tois[i][1], pred_cls[i]) for i in range(len(tois))]

        for pre in pred_entities:
            if pre not in gold_entities:
                dic["FP"][pre[2] - 1] += 1
        for gt in gold_entities:
            if gt not in pred_entities:
                dic["FN"][gt[2] - 1] += 1

    def dfs(self, idx, gt, gt_layer):
        if gt_layer[idx] != -1:
            return gt_layer[idx];
        gt_layer[idx] = -2;

        for contain_id in range(len(gt)):
            if contain_id == idx:
                continue
            if gt[contain_id][0] >= gt[idx][0] and gt[contain_id][1] <= gt[idx][1] and gt_layer[contain_id] != -2:
                gt_layer[idx] = max(gt_layer[idx], self.dfs(contain_id, gt, gt_layer) + 1)

        if gt_layer[idx] == -2:
            gt_layer[idx] = 0;

        self.max_layer = max(self.max_layer, gt_layer[idx]);
        return gt_layer[idx];

    def evaluate_detail(self, gold_entities, pred_entities, pred_depth):
        gold_entities = list(gold_entities)
        if(len(gold_entities)):
            gt_layer = [-1] * len(gold_entities)
        for id in range(len(gold_entities)):
            if gt_layer[id] == -1:
                self.dfs(id, gold_entities, gt_layer)

        for id in range(len(gold_entities)):
            if gt_layer[id] >= 1:
                gt_layer[id] = 1
                gold_entities[id] = list(gold_entities[id])
                gold_entities[id] = tuple(gold_entities[id])

        for idx in range(len(gold_entities)):
            if gold_entities[idx] not in pred_entities:
                self.confusion_matrix[gold_entities[idx][2] - 1, self.classes_num] += 1  # FN

        right_pres = []
        for idx in range(len(pred_entities)):
            if tuple(pred_entities[idx]) in gold_entities :
                right_pres.append(pred_entities[idx])
                self.confusion_matrix[pred_entities[idx][2] - 1, pred_entities[idx][2] - 1] += 1   # TP
                if pred_entities[idx][2] > 6:
                    self.layer_precision[1][1] += 1
                else:
                    self.layer_precision[0][0] += 1

            else:
                self.confusion_matrix[self.classes_num, pred_entities[idx][2] - 1] += 1            # FP
                if pred_entities[idx][2] > 6:
                    self.layer_precision[1][2] += 1
                else:
                    self.layer_precision[0][2] += 1

        for i in range(len(right_pres)):
            for j in range(i + 1, len(right_pres)):
                if check_contain(right_pres[i][0:2], right_pres[j][0:2]):
                    self.contain_matrix[right_pres[i][2] - 1, right_pres[j][2] - 1] += 1
                elif check_contain(right_pres[j][0:2], right_pres[i][0:2]):
                    self.contain_matrix[right_pres[j][2] - 1, right_pres[i][2] - 1] += 1

    def getId(self, a, b):
        for i in range(len(b)):
            if tuple(a) == b[i]:
                return i

    def evaluate(self, gold_entities, pred_entities):
        self.recall_all += len(gold_entities)
        self.pred_all += len(pred_entities)

        for gt in gold_entities:
            if gt in pred_entities:
                self.recall += 1

        for pre in pred_entities:
            if pre in gold_entities:
                self.pred += 1

    def init(self):
        self.confusion_matrix = np.zeros((self.classes_num + 1, self.classes_num + 1))
        self.layer_precision = np.zeros((2, 3))
        self.contain_matrix = np.zeros((self.classes_num, self.classes_num))
        self.toi_F = {"FN": [0] * self.classes_num, "FP": [0] * self.classes_num}
        self.len_F = {"FN": [0] * self.classes_num, "FP": [0] * self.classes_num}
        self.score_F = {"FN": [0] * self.classes_num, "FP": [0] * self.classes_num}
        self.max_layer = 0
        self.pred_all, self.pred, self.recall_all, self.recall = 0, 0, 0, 0

    def get_count_dataframe(self, confusion_matrix, labels):
        num = len(labels)
        experiment_metrics = []
        sum_result_dir = {"TP": 0, "FP": 0, "FN": 0}
        cnt = 0
        for one_label in range(num):
            TP = confusion_matrix[one_label, one_label]
            sum_result_dir["TP"] += TP
            FP = sum(confusion_matrix[:, one_label]) - TP
            sum_result_dir["FP"] += FP
            FN = sum(confusion_matrix[one_label, :]) - TP
            sum_result_dir["FN"] += FN

            precision = TP / (TP + FP) if TP + FP != 0 else 0
            recall = TP / (TP + FN) if TP + FN != 0 else 0
            F1_score = (2 * precision * recall) / (precision + recall) if precision * recall != 0 else 0
            experiment_metrics.append([precision, recall, F1_score])
            cnt += 1
            if(cnt == num // 2):
                TP = sum_result_dir["TP"]
                FP = sum_result_dir["FP"]
                FN = sum_result_dir["FN"]
                precision = TP / (TP + FP) if TP + FP != 0 else 0
                recall = TP / (TP + FN) if TP + FN != 0 else 0
                F1_score = (2 * precision * recall) / (precision + recall) if precision * recall != 0 else 0
                experiment_metrics.append([precision, recall, F1_score])

        TP = sum_result_dir["TP"]
        FP = sum_result_dir["FP"]
        FN = sum_result_dir["FN"]
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        F1_score = (2 * precision * recall) / (precision + recall) if precision * recall != 0 else 0

        experiment_metrics.append([precision, recall, F1_score])
        return pd.DataFrame(experiment_metrics, columns=["precision", "recall", "F1_score"],
                            index=labels[0:num // 2] + ['overall-l1'] + labels[num//2:] + ["overall"]), F1_score
