import gc
import yaml
import json
import torch
from transformers import GenerationConfig
from models import alpaca, stablelm, koalpaca, flan_alpaca, mpt
from models import camel, t5_vicuna, vicuna, starchat, redpajama, bloom
from models import baize, guanaco, falcon, kullm, replit, airoboros
from models import samantha_vicuna

from utils import get_chat_interface, get_chat_manager

model_infos = json.load(open("model_cards.json"))

def get_model_type(model_info):
    base_url = model_info["hub(base)"]
    ft_ckpt_url = model_info["hub(ckpt)"]
    
    model_type_tmp = "alpaca"
    if "llms/wizardlm" in base_url.lower():
        model_type_tmp = "wizardlm"
    elif "chronos" in base_url.lower():
        model_type_tmp = "chronos"
    elif "lazarus" in base_url.lower():
        model_type_tmp = "lazarus"
    elif "samantha" in base_url.lower():
        model_type_tmp = "samantha-vicuna"
    elif "airoboros" in base_url.lower():
        model_type_tmp = "airoboros"
    elif "replit" in base_url.lower():
        model_type_tmp = "replit-instruct"
    elif "kullm" in base_url.lower():
        model_type_tmp = "kullm-polyglot"
    elif "nous-hermes" in base_url.lower():
        model_type_tmp = "nous-hermes"
    elif "guanaco" in base_url.lower():
        model_type_tmp = "guanaco"
    elif "wizardlm-uncensored-falcon" in base_url.lower():
        model_type_tmp = "wizard-falcon"        
    elif "falcon" in base_url.lower():
        model_type_tmp = "falcon"
    elif "baize" in base_url.lower():
        model_type_tmp = "baize"
    elif "stable-vicuna" in base_url.lower():
        model_type_tmp = "stable-vicuna"        
    elif "vicuna" in base_url.lower():
        model_type_tmp = "vicuna"
    elif "mpt" in base_url.lower():
        model_type_tmp = "mpt"
    elif "redpajama-incite-7b-instruct" in base_url.lower():
        model_type_tmp = "redpajama-instruct"
    elif "redpajama" in base_url.lower():
        model_type_tmp = "redpajama"
    elif "starchat" in base_url.lower():
        model_type_tmp = "starchat"
    elif "camel" in base_url.lower():
        model_type_tmp = "camel"
    elif "flan-alpaca" in base_url.lower():
        model_type_tmp = "flan-alpaca"
    elif "openassistant/stablelm" in base_url.lower():
        model_type_tmp = "os-stablelm"
    elif "stablelm" in base_url.lower():
        model_type_tmp = "stablelm"
    elif "fastchat-t5" in base_url.lower():
        model_type_tmp = "t5-vicuna"
    elif "koalpaca-polyglot" in base_url.lower():
        model_type_tmp = "koalpaca-polyglot"
    elif "alpacagpt4" in ft_ckpt_url.lower():
        model_type_tmp = "alpaca-gpt4"
    elif "alpaca" in ft_ckpt_url.lower():
        model_type_tmp = "alpaca"
    elif "llama-deus" in ft_ckpt_url.lower():
        model_type_tmp = "llama-deus"
    elif "vicuna-lora-evolinstruct" in ft_ckpt_url.lower():
        model_type_tmp = "evolinstruct-vicuna"
    elif "alpacoom" in ft_ckpt_url.lower():
        model_type_tmp = "alpacoom"
    elif "guanaco" in ft_ckpt_url.lower():
        model_type_tmp = "guanaco"
    else:
        print("unsupported model type")
        
    return model_type_tmp

def initialize_globals():
    global models, tokenizers
    
    models = []
    model_names = [
        "baize-7b",   
        # "llama-deus-7b",
        "evolinstruct-vicuna-13b",
        # "koalpaca",
        "guanaco-13b",
        "nous-hermes-13b"
    ]
    for model_name in model_names:
        model_info = model_infos[model_name]
        model_thumbnail_tiny = model_info["thumb-tiny"]
        model_type = get_model_type(model_info)
        print(model_type)
        load_model = get_load_model(model_type)
        
        model, tokenizer = load_model(
            base=model_info["hub(base)"],
            finetuned=model_info["hub(ckpt)"],
            mode_cpu=False,
            mode_mps=False,
            mode_full_gpu=True,
            mode_8bit=False,
            mode_4bit=False,
            force_download_ckpt=False
        )        
        
        gen_config, gen_config_raw = get_generation_config(
            model_info["default_gen_config"]
        )
        
        models.append(
            {
                "model_name": model_name,
                "model_thumb_tiny": model_thumbnail_tiny,
                "model_type": model_type,
                "model": model,
                "tokenizer": tokenizer,
                "gen_config": gen_config,
                "gen_config_raw": gen_config_raw,
                "chat_interface": get_chat_interface(model_type),
                "chat_manager": get_chat_manager(model_type),
            }
        )
        
def get_load_model(model_type):
    if model_type == "alpaca" or \
        model_type == "alpaca-gpt4" or \
        model_type == "llama-deus" or \
        model_type == "nous-hermes" or \
        model_type == "lazarus" or \
        model_type == "chronos" or \
        model_type == "wizardlm":
        return alpaca.load_model
    elif model_type == "stablelm" or model_type == "os-stablelm":
        return stablelm.load_model
    elif model_type == "koalpaca-polyglot":
        return koalpaca.load_model
    elif model_type == "kullm-polyglot":
        return kullm.load_model
    elif model_type == "flan-alpaca":
        return flan_alpaca.load_model
    elif model_type == "camel":
        return camel.load_model
    elif model_type == "t5-vicuna":
        return t5_vicuna.load_model
    elif model_type == "stable-vicuna":
        return vicuna.load_model
    elif model_type == "starchat":
        return starchat.load_model
    elif model_type == "mpt":
        return mpt.load_model
    elif model_type == "redpajama" or \
        model_type == "redpajama-instruct":
        return redpajama.load_model
    elif model_type == "vicuna":
        return vicuna.load_model
    elif model_type == "evolinstruct-vicuna":
        return alpaca.load_model
    elif model_type == "alpacoom":
        return bloom.load_model
    elif model_type == "baize":
        return baize.load_model
    elif model_type == "guanaco":
        return guanaco.load_model
    elif model_type == "falcon" or model_type == "wizard-falcon":
        return falcon.load_model
    elif model_type == "replit-instruct":
        return replit.load_model
    elif model_type == "airoboros":
        return airoboros.load_model
    elif model_type == "samantha-vicuna":
        return samantha_vicuna.load_model
    else:
        return None
    
def get_generation_config(path):
    with open(path, 'rb') as f:
        generation_config = yaml.safe_load(f.read())
        
    generation_config = generation_config["generation_config"]

    return GenerationConfig(**generation_config), generation_config

def get_constraints_config(path):
    with open(path, 'rb') as f:
        constraints_config = yaml.safe_load(f.read())
        
    return ConstraintsConfig(**constraints_config), constraints_config["constraints"]
