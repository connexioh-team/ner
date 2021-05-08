from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import regex
import json
import pickle
import torch
from gluonnlp.data import SentencepieceTokenizer
from model.net import KobertCRF
from data_utils.utils import Config
from data_utils.vocab_tokenizer import Tokenizer
from data_utils.pad_sequence import keras_pad_fn
from pathlib import Path

class DecoderFromNamedEntitySequence():
    def __init__(self, tokenizer, index_to_ner):
        self.tokenizer = tokenizer
        self.index_to_ner = index_to_ner

    def __call__(self, list_of_input_ids, list_of_pred_ids):
        input_token = self.tokenizer.decode_token_ids(list_of_input_ids)[0]
        pred_ner_tag = [self.index_to_ner[pred_id] for pred_id in list_of_pred_ids[0]]

        # ----------------------------- parsing list_of_ner_word ----------------------------- #
        list_of_ner_word = []
        entity_word, entity_tag, prev_entity_tag = "", "", ""
        for i, pred_ner_tag_str in enumerate(pred_ner_tag):
            if "B-" in pred_ner_tag_str:
                entity_tag = pred_ner_tag_str[-3:]

                if prev_entity_tag != entity_tag and prev_entity_tag != "":
                    list_of_ner_word.append(regex.sub("[^\p{L}\p{M}\p{N}\s]+", "", entity_word) + "/" + prev_entity_tag)

                entity_word = input_token[i]
                prev_entity_tag = entity_tag
            elif "I-"+entity_tag in pred_ner_tag_str:
                entity_word += input_token[i]
            else:
                if entity_word != "" and entity_tag != "":
                    list_of_ner_word.append(regex.sub("[^\p{L}\p{M}\p{N}\s]+", "", entity_word) + "/" + entity_tag)
                entity_word, entity_tag, prev_entity_tag = "", "", ""

        return list_of_ner_word

def main():
    cur_path = os.path.dirname(sys.argv[0])
    if cur_path:
        print(cur_path)
        os.chdir(cur_path)

    model_dir = Path('./experiments/base_model_with_crf')
    model_config = Config(json_path=model_dir / 'config.json')

    # load vocab & tokenizer
    tok_path = "./ptr_lm_model/tokenizer_78b3253a26.model"
    ptr_tokenizer = SentencepieceTokenizer(tok_path)

    with open(model_dir / "vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)

    # load ner_to_index.json
    with open(model_dir / "ner_to_index.json", 'rb') as f:
        ner_to_index = json.load(f)
        index_to_ner = {v: k for k, v in ner_to_index.items()}

    # model
    model = KobertCRF(config=model_config, num_classes=len(ner_to_index), vocab=vocab)

    # load
    model_dict = model.state_dict()
    checkpoint = torch.load("./best-epoch-16-step-1500-acc-0.993.bin", map_location=torch.device('cpu'))
    #checkpoint = torch.load("./experiments/base_model_with_crf/best-epoch-16-step-1500-acc-0.993.bin", map_location=torch.device('cpu'))
    # checkpoint = torch.load("./experiments/base_model_with_crf_val/best-epoch-12-step-1000-acc-0.960.bin", map_location=torch.device('cpu'))
    convert_keys = {}
    for k, v in checkpoint['model_state_dict'].items():
        new_key_name = k.replace("module.", '')
        if new_key_name not in model_dict:
            print("{} is not int model_dict".format(new_key_name))
            continue
        convert_keys[new_key_name] = v

    model.load_state_dict(convert_keys, strict=False)
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #model.to(device)
    decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_ner)
    
    try:
        while(True):
            input_text = input()
        
            list_of_input_ids = tokenizer.list_of_string_to_list_of_cls_sep_token_ids([input_text])
            x_input = torch.tensor(list_of_input_ids).long()
            list_of_pred_ids = model(x_input)

            list_of_ner_word = decoder_from_res(list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids)
            print(list_of_ner_word)
            print("")
    except:
        print("EOF")


if __name__ == "__main__":
    main()
