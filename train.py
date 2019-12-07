import pickle
import random
import time
import numpy as np
import torch
from random import shuffle
from Evaluate import Evaluate
from config import config
from model import TOI_BERT
from utils import create_opt, select_toi, adjust_learning_rate, generate_mask, dfs

from get_model_result import ModelInRuntime
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from tqdm import tqdm
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(all_batch=[], read_from_file=True, section=[]):
    if read_from_file:
        with open(config.get_pkl_path("train"), "rb") as f:
            train_word_batches, train_char_batches, train_char_len_batches, train_pos_tag_batches, train_entity_batches, train_toi_batches, train_word_origin_batches = pickle.load(f)
        with open(config.get_pkl_path("test"), "rb") as f:
            test_word_batches, test_char_batches, test_char_len_batches, test_pos_tag_batches, test_entity_batches, test_toi_batches, test_word_origin_batches = pickle.load(f)
    else:
        train_word_batches, train_char_batches, train_char_len_batches, train_pos_tag_batches, train_entity_batches, train_toi_batches, train_toi_batch_layer0, train_toi_batch_layer1 = all_batch[0]
        dev_word_batches, dev_char_batches, dev_char_len_batches, dev_pos_tag_batches, dev_entity_batches, dev_toi_batches, dev_toi_batch_layer0, dev_toi_batch_layer1 = all_batch[1]
        test_word_batches, test_char_batches, test_char_len_batches, test_pos_tag_batches, test_entity_batches, test_toi_batches, test_toi_batch_layer0, test_toi_batch_layer1 = all_batch[2]

    misc_config = pickle.load(open(config.get_pkl_path("config"), "rb"))
    config.load_config(misc_config)

    ner_model = TOI_BERT(config)
    if config.if_DTE:
        ner_model.load_vector()

    if(len(section)):
        config.layer_maxlen = section

    if config.if_gpu and torch.cuda.is_available():
        ner_model = ner_model.cuda()

    evaluate = Evaluate(ner_model, config)

    parameters = filter(lambda p: p.requires_grad, ner_model.parameters())
    optimizer = create_opt(parameters, config.opt, config.lr)

    best_model = None
    best_per = 0
    pre_loss = 100000
    train_all_batches = list(zip(train_word_batches, train_char_batches, train_char_len_batches, train_pos_tag_batches, train_entity_batches, train_toi_batches, train_word_origin_batches))
    tokenizer = BertTokenizer.from_pretrained(f"bert-{config.bert_config}-uncased")
    bert_model = BertModel.from_pretrained(f"{config.bert_path}{config.bert_config}")
    bert_model.cuda()
    bert_model.eval()
    for parameter in bert_model.parameters():
        parameter.requires_grad = False

    for e_ in range(config.epoch):
        print("Epoch:", e_ + 1)

        cur_time = time.time()
        if config.if_shuffle:
            shuffle(train_all_batches)
        losses = []
        ner_model.train()
        config.mode = 'Train'
        runtimeModel = ModelInRuntime.instance((ner_model, bert_model, tokenizer, config, len(train_all_batches) + len(test_word_batches)))
        runtimeModel.model = ner_model

        for each_batch in tqdm(train_all_batches):
            optimizer.zero_grad()
            runtimeModel.setTrainData(each_batch)
            result, _, aim = runtimeModel.runClassification()
            loss = ner_model.calc_loss(result, aim)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu().numpy())

        sub_loss = np.mean(losses)
        print(f'Avg loss = {sub_loss:.4f}')
        print(f"Training step took {time.time() - cur_time:.0f} seconds")
        if e_ >= 0:
            print("dev:")
            cls_f1 = evaluate.get_f1(zip(test_word_batches, test_char_batches, test_char_len_batches, test_pos_tag_batches, test_entity_batches, test_toi_batches, test_word_origin_batches), bert_model)
            if cls_f1 > best_per and cls_f1 > config.score_th:
                best_per = cls_f1
                model_path = config.get_model_path() + f"/epoch{e_ + 1}_f1_{cls_f1:.4f}.pth"
                torch.save(ner_model.state_dict(), model_path)
                print("model save in " + model_path)
            print('\n\n')
        if sub_loss >= pre_loss:
            adjust_learning_rate(optimizer)
        pre_loss = sub_loss


if __name__ == '__main__':
    setup_seed(12345)
    train()
