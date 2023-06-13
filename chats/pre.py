import re
import copy
import global_vars
from threading import Thread
from transformers import TextIteratorStreamer
from transformers import GenerationConfig

def contains_image_markdown(string):
    regex = re.compile(r'!\[(.*?)\]\((.*?)\)')
    match = regex.search(string)
    return match

def build_model_inputs(prompt, model_num, return_token_type_ids):
    model_inputs = global_vars.models[model_num]["tokenizer"](
        [prompt], 
        return_tensors="pt",
        return_token_type_ids=return_token_type_ids
    ).to("cuda")
    return model_inputs

def build_streamer(
    model_num,
    timeout=20.,
    skip_prompt=True,
    skip_special_tokens=True
):
    streamer = TextIteratorStreamer(
        global_vars.models[model_num]["tokenizer"], 
        timeout=timeout, 
        skip_prompt=skip_prompt,
        skip_special_tokens=skip_special_tokens
    )
    return streamer


def build_gen_config(
    temperature, top_p, top_k, repetition_penalty, max_new_tokens, 
    num_beams, use_cache, do_sample, eos_token_id, pad_token_id 
):
    gen_config_raw = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "use_cache": use_cache,
        "do_sample": do_sample,
        "eos_token_id": eos_token_id, 
        "pad_token_id": pad_token_id
    }

    return gen_config_raw, GenerationConfig(**gen_config_raw)

def build_gen_kwargs(
    gen_config,
    model_inputs,
    streamer,
    stopping_criteria
):
    gen_kwargs = dict(
        model_inputs,
        streamer=streamer,
        stopping_criteria=stopping_criteria
    )
    gen_kwargs.update(gen_config)
    return gen_kwargs 

def start_gen(gen_kwargs, model_num):
    t = Thread(
        target=global_vars.models[model_num]["model"].generate,
        kwargs=gen_kwargs
    )
    t.start()

def build(
    prompt, model_num,
    temperature, top_p, top_k, repetition_penalty, max_new_tokens, 
    num_beams, use_cache, do_sample, eos_token_id, pad_token_id,
    stopping_criteria=None, return_token_type_ids=True
):
    gen_config_raw, _ = build_gen_config(
        temperature, top_p, top_k, repetition_penalty, max_new_tokens, 
        num_beams, use_cache, do_sample, eos_token_id, pad_token_id 
    )

    model_inputs = build_model_inputs(
        prompt, model_num, return_token_type_ids=return_token_type_ids
    )
    streamer = build_streamer(model_num)
    gen_kwargs = build_gen_kwargs(
        gen_config_raw, 
        model_inputs, 
        streamer,
        stopping_criteria
    )
    return gen_kwargs, streamer