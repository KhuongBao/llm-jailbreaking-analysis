import torch
from utils import load_model, SAVE_PATH, DEVICE



### ------------- Save path ----------------------------
import os
os.makedirs(SAVE_PATH, exist_ok=True)

uncensored_attacker_dir = os.path.join(SAVE_PATH, "uncensored_attacker_model")
censored_attacker_dir = os.path.join(SAVE_PATH, "censored_attacker_model")
target_dir = os.path.join(SAVE_PATH, "target_model")
cls_dir = os.path.join(SAVE_PATH, "cls_model")



### ------------- Loading Models -----------------------

def import_models():
    uncensored_attacker_id = "failspy/Llama-3-8B-Instruct-Abliterated"
    uncensored_attacker_tokenizer, uncensored_attacker_model = load_model(uncensored_attacker_id, DEVICE)

    censored_attacker_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    censored_attacker_tokenizer, censored_attacker_model = load_model(censored_attacker_id, DEVICE)

    target_id = "google/gemma-2-9b-it"
    target_tokenizer, target_model = load_model(target_id, DEVICE)

    cls_id = "cais/HarmBench-Mistral-7b-val-cls"
    cls_tokenizer, cls_model = load_model(cls_id, DEVICE)

    return uncensored_attacker_tokenizer, uncensored_attacker_model, \
            censored_attacker_tokenizer, censored_attacker_model,    \
            target_tokenizer, target_model,                          \
            cls_tokenizer, cls_model 