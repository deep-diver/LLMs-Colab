from chats import stablelm
from chats import alpaca
from chats import koalpaca
from chats import flan_alpaca
from chats import os_stablelm
from chats import vicuna
from chats import starchat
from chats import redpajama
from chats import mpt
from chats import alpacoom
from chats import baize
from chats import guanaco

def chat_stream(
    idx, local_data, user_message, state, model_num,
    global_context, ctx_num_lconv, ctx_sum_prompt,
    res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
):
    model_type = state["model_type"]

    if model_type == "stablelm":
        cs = stablelm.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )

    elif model_type == "baize":
        cs = baize.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )        
        
    elif model_type == "alpaca":
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )

    elif model_type == "alpaca-gpt4":
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )

    elif model_type == "alpacoom":
        cs = alpacoom.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )        

    elif model_type == "llama-deus":
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )

    elif model_type == "camel":
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )

    elif model_type == "koalpaca-polyglot":
        cs = koalpaca.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )

    elif model_type == "flan-alpaca":
        cs = flan_alpaca.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )        

    elif model_type == "os-stablelm":
        cs = os_stablelm.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )

    elif model_type == "t5-vicuna":
        cs = vicuna.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )

    elif model_type == "stable-vicuna":
        cs = vicuna.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )

    elif model_type == "vicuna":
        cs = vicuna.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )        

    elif model_type == "evolinstruct-vicuna":
        cs = vicuna.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )

    elif model_type == "starchat":
        cs = starchat.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )

    elif model_type == "mpt":
        cs = mpt.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )

    elif model_type == "redpajama":
        cs = redpajama.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )

    elif model_type == "guanaco":
        cs = guanaco.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )

    elif model_type == "nous-hermes":
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state, model_num,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        )        
        
    for idx, x in enumerate(cs):
        yield x        
        