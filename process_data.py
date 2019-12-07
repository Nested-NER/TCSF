import pickle

from config import config
from Reader import Reader
from transfer_wv import work

# Convert word  token to embeddings.

work()
class preprocess_data():
    def __init__(self, length=config.Lb, save=True, max_length=[]):
        if_save = save
        reader = Reader(config)
        reader.read_all_data()
        self.ret_batch = []
        config.Lb = length
        for mode in ["train", "dev", "test"]:
            all_batches = reader.to_batch(mode)
            print(mode + ":")
            print(f"max entity len in {mode}: {reader.max_C[mode]}")
            print(f"num of sent in {mode}: {len(reader.infos[mode])}")
            print(f"num of the pos in {mode}: {reader.candidation_hit[mode]}")
            print(f"num of the neg in {mode}: {reader.tois_num[mode] - reader.candidation_hit[mode]}")
            print(f"candidation recall in {mode}: {reader.candidation_hit[mode] / reader.entity_num[mode]:.4f}")

            if if_save:
                with open(reader.config.get_pkl_path(mode), "wb") as f:
                    pickle.dump(all_batches, f)
            else:
                self.ret_batch.append(all_batches)

        misc_dict = dict()
        misc_dict["word_kinds"] = len(reader.id2word)
        misc_dict["char_kinds"] = len(reader.char2id)
        misc_dict["pos_tag_kinds"] = len(reader.pos_tag2id)
        misc_dict["label_kinds"] = len(reader.label2id)
        misc_dict["id2label"] = reader.id2label
        misc_dict["id2word"] = reader.id2word
        if if_save:
            path = reader.config.get_pkl_path("config")
            with open(path, "wb") as f:
                pickle.dump(misc_dict, f)
                print("config saving in ", path)


if __name__ == '__main__':
    a = preprocess_data()
