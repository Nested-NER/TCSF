"""Could test best model provided by us, in for loop."""
import pickle
import torch
import os
from Evaluate import Evaluate
from config import config
from model import TOI_BERT
from pytorch_pretrained_bert import BertModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

mode = "test"  # test dev
test_best = True
epoch_start = 1
epoch_end = 100

misc_config = pickle.load(open(config.get_pkl_path("config"), "rb"))
config.load_config(misc_config)
bert_model = BertModel.from_pretrained(f"{config.bert_path}{config.bert_config}")
bert_model.cuda()

bert_model.eval()

model_path = config.get_model_path() + "f1_0.771.pth"

with open(config.get_pkl_path(mode), "rb") as f:
    word_batches, char_batches, char_len_batches, pos_tag_batches, entity_batches, toi_batches,  word_origin_batches = pickle.load(f)
print("load data from " + config.get_pkl_path(mode))

#print(model_path)
if not os.path.exists(model_path):
   print("loda model error")
print("load model from " + model_path)
model1 = torch.load(model_path)
ner_model = TOI_BERT(config)
ner_model.load_state_dict(torch.load(model_path))
if config.if_gpu and torch.cuda.is_available():
    ner_model = ner_model.cuda()
evaluate = Evaluate(ner_model, config)
evaluate.get_f1(zip(word_batches, char_batches, char_len_batches, pos_tag_batches, entity_batches, toi_batches, word_origin_batches), bert_model)
print("\n\n")
