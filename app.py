import time
import json
from os import listdir
from os.path import isfile, join
import gradio as gr
import args
import global_vars
from chats import central
from transformers import AutoModelForCausalLM
from miscs.styles import MODEL_SELECTION_CSS
from miscs.js import GET_LOCAL_STORAGE, UPDATE_LEFT_BTNS_STATE
from utils import get_chat_interface, get_chat_manager

ex_file = open("examples.txt", "r")
examples = ex_file.read().split("\n")
ex_btns = []

chl_file = open("channels.txt", "r")
channels = chl_file.read().split("\n")
channel_btns = []

global_vars.initialize_globals()

response_configs = [
    f"configs/response_configs/{f}"
    for f in listdir("configs/response_configs")
    if isfile(join("configs/response_configs", f))
]

summarization_configs = [
    f"configs/summarization_configs/{f}"
    for f in listdir("configs/summarization_configs")
    if isfile(join("configs/summarization_configs", f))
]

model_info = json.load(open("model_cards.json"))

def channel_num(btn_title):
    choice = 0

    for idx, channel in enumerate(channels):
        if channel == btn_title:
            choice = idx

    return choice


def set_chatbot(btn, ld, state):
    choice = channel_num(btn)

    res = [state["ppmanager_type"].from_json(json.dumps(ppm_str)) for ppm_str in ld]
    empty = len(res[choice].pingpongs) == 0
    return (
        res[choice].build_uis(), 
        choice, 
        gr.update(visible=empty), 
        gr.update(interactive=not empty)
    )


def set_example(btn):
    return btn, gr.update(visible=False)


def set_popup_visibility(ld, example_block):
    return example_block


def move_to_second_view(btn):
    info = model_info[btn]

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        info["thumb"],
        f"**Model name**\n: {btn}",
        f"**Parameters**\n: {info['parameters']}",
        f"**🤗 Hub(base)**\n: {info['hub(base)']}",
        f"**🤗 Hub(ckpt)**\n: {info['hub(ckpt)']}",
        "",
    )


def move_to_first_view():
    return (gr.update(visible=True), gr.update(visible=False))


def get_model_num(
    model_name
):
    model_num = 0    
    model_name = model_name.split(":")[-1].split("</p")[0].strip()
    
    for idx, item in enumerate(global_vars.models):
        if item["model_name"] == model_name:
            model_num = idx
            break
            
    return "Download completed!", model_num

def move_to_third_view(model_num):
    gen_config = global_vars.models[model_num]["gen_config"]

    return (
        "Preparation done!",
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(label=global_vars.models[model_num]["model_type"]),
        {
            "ppmanager_type": global_vars.models[model_num]["chat_manager"],
            "model_type": global_vars.models[model_num]["model_type"],
        },
        gen_config.temperature,
        gen_config.top_p,
        gen_config.top_k,
        gen_config.repetition_penalty,
        gen_config.max_new_tokens,
        gen_config.num_beams,
        gen_config.use_cache,
        gen_config.do_sample,
        gen_config.eos_token_id,
        gen_config.pad_token_id,
    )


def toggle_inspector(view_selector):
    if view_selector == "with context inspector":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def reset_chat(idx, ld, state):
    res = [state["ppmanager_type"].from_json(json.dumps(ppm_str)) for ppm_str in ld]
    res[idx].pingpongs = []
        
    return (
        "",
        [],
        str(res),
        gr.update(visible=True),
        gr.update(interactive=False),
    )

def rollback_last(idx, ld, state):
    res = [state["ppmanager_type"].from_json(json.dumps(ppm_str)) for ppm_str in ld]
    last_user_message = res[idx].pingpongs[-1].ping
    res[idx].pingpongs = res[idx].pingpongs[:-1]
    
    return (
        last_user_message,
        res[idx].build_uis(),
        str(res),
        gr.update(interactive=False)
    )

with gr.Blocks(css=MODEL_SELECTION_CSS, theme='gradio/soft') as demo:
    with gr.Column() as model_choice_view:
        gr.Markdown("# Choose a Model", elem_classes=["center"])
        gr.Markdown("### NOTE")
        gr.Markdown(":If you want to re-select a model after entering the chat mode, you can do it by refreshing the web page. Sorry for the inconvenience for now, but it will be replaced with a better UI/UX soon")
        with gr.Row(elem_id="container"):
            with gr.Column():
                with gr.Row():
                    with gr.Column(min_width=20):
                        llama_deus_7b = gr.Button(
                            "llama-deus-7b",
                            elem_id="llama-deus-7b",
                            elem_classes=["square"],
                        )
                        gr.Markdown("LLaMA Deus", elem_classes=["center"])                    

                    with gr.Column(min_width=20):                        
                        baize_7b = gr.Button(
                            "baize-7b",
                            elem_id="baize-7b",
                            elem_classes=["square"],
                        )
                        gr.Markdown("Baize", elem_classes=["center"])                            
                        
                    with gr.Column(min_width=20):
                        koalpaca = gr.Button(
                            "koalpaca", elem_id="koalpaca", elem_classes=["square"]
                        )
                        gr.Markdown("koalpaca", elem_classes=["center"])                        
                        
                    with gr.Column(min_width=20):
                        evolinstruct_vicuna_13b = gr.Button(
                            "evolinstruct-vicuna-13b",
                            elem_id="evolinstruct-vicuna-13b",
                            elem_classes=["square"],
                        )
                        gr.Markdown("EvolInstruct Vicuna", elem_classes=["center"])                      
                        
                    with gr.Column(min_width=20):
                        guanaco_33b = gr.Button(
                            "guanaco-33b", elem_id="guanaco-33b", elem_classes=["square"]
                        )
                        gr.Markdown("Guanaco", elem_classes=["center"])
                        
                progress_view = gr.Textbox(label="Progress")

    with gr.Column(visible=False) as model_review_view:
        gr.Markdown("# Confirm the chosen model", elem_classes=["center"])
        with gr.Column(elem_id="container2"):
            with gr.Row():
                model_image = gr.Image(None, interactive=False, show_label=False)
                with gr.Column():
                    model_name = gr.Markdown("**Model name**")
                    model_params = gr.Markdown("Parameters\n: ...")
                    model_base = gr.Markdown("🤗 Hub(base)\n: ...")
                    model_ckpt = gr.Markdown("🤗 Hub(ckpt)\n: ...")

            with gr.Row():
                back_to_model_choose_btn = gr.Button("Back")
                confirm_btn = gr.Button("Confirm")

            progress_view2 = gr.Textbox(label="Progress")                
                
    with gr.Column(visible=False) as chat_view:
        idx = gr.State(0)
        model_num = gr.State(0)
        chat_state = gr.State()
        local_data = gr.JSON({}, visible=False)

        gr.Markdown("### NOTE")
        gr.Markdown(":If you want to re-select a model, you need to refresh the web page. \
                            Sorry for the inconvenience for now, but it will be replaced with a better UI/UX soon.")
        gr.Markdown(":The chat histories are kept in your local browser, so you will be able to play with different models \
                            while keeping the records of the previous conversations")
        
        with gr.Row():
            with gr.Column(scale=1, min_width=180):
                gr.Markdown("GradioChat", elem_id="left-top")

                with gr.Column(elem_id="left-pane"):
                    with gr.Accordion("Histories", elem_id="chat-history-accordion"):
                        channel_btns.append(gr.Button(channels[0], elem_classes=["custom-btn-highlight"]))

                        for channel in channels[1:]:
                            channel_btns.append(gr.Button(channel, elem_classes=["custom-btn"]))

            with gr.Column(scale=8, elem_id="right-pane"):
                with gr.Column(
                    elem_id="initial-popup", visible=False
                ) as example_block:
                    with gr.Row(scale=1):
                        with gr.Column(elem_id="initial-popup-left-pane"):
                            gr.Markdown("GradioChat", elem_id="initial-popup-title")
                            gr.Markdown(
                                "Making the community's best AI chat models available to everyone."
                            )
                        with gr.Column(elem_id="initial-popup-right-pane"):
                            gr.Markdown(
                                "Chat UI is now open sourced on Hugging Face Hub"
                            )
                            gr.Markdown(
                                "check out the [↗ repository](https://huggingface.co/spaces/chansung/test-multi-conv)"
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("Examples")
                        with gr.Row():
                            for example in examples:
                                ex_btns.append(gr.Button(example, elem_classes=["example-btn"]))

                with gr.Column(elem_id="aux-btns-popup", visible=True):
                    with gr.Row():
                        stop = gr.Button("Stop", elem_classes=["aux-btn"], interactive=False)
                        regenerate = gr.Button("Regenerate", interactive=False, elem_classes=["aux-btn"])
                        clean = gr.Button("Clean", elem_classes=["aux-btn"])

                with gr.Accordion("Context Inspector", elem_id="aux-viewer", open=False):
                    context_inspector = gr.Textbox(
                        "",
                        elem_id="aux-viewer-inspector",
                        label="",
                        lines=30,
                        max_lines=50,
                    )                        
                        
                chatbot = gr.Chatbot(elem_id='chatbot')
                instruction_txtbox = gr.Textbox(
                    placeholder="Ask anything", label="",
                    elem_id="prompt-txt"
                )

        with gr.Accordion("Constrol Panel", open=False) as control_panel:
            with gr.Column():
                with gr.Column():
                    gr.Markdown("#### GenConfig for **response** text generation")
                    with gr.Row():
                        res_temp = gr.Slider(0.0, 2.0, 0, step=0.1, label="temp", interactive=True)
                        res_topp = gr.Slider(0.0, 2.0, 0, step=0.1, label="top_p", interactive=True)
                        res_topk = gr.Slider(20, 1000, 0, step=1, label="top_k", interactive=True)
                        res_rpen = gr.Slider(0.0, 2.0, 0, step=0.1, label="rep_penalty", interactive=True)
                        res_mnts = gr.Slider(64, 2048, 0, step=1, label="new_tokens", interactive=True)                            
                        res_beams = gr.Slider(1, 4, 0, step=1, label="beams")
                        res_cache = gr.Radio([True, False], value=0, label="cache", interactive=True)
                        res_sample = gr.Radio([True, False], value=0, label="sample", interactive=True)
                        res_eosid = gr.Number(value=0, visible=False, precision=0)
                        res_padid = gr.Number(value=0, visible=False, precision=0)

                with gr.Column(visible=False):
                    gr.Markdown("#### GenConfig for **summary** text generation")
                    with gr.Row():
                        sum_temp = gr.Slider(0.0, 2.0, 0, step=0.1, label="temp", interactive=True)
                        sum_topp = gr.Slider(0.0, 2.0, 0, step=0.1, label="top_p", interactive=True)
                        sum_topk = gr.Slider(20, 1000, 0, step=1, label="top_k", interactive=True)
                        sum_rpen = gr.Slider(0.0, 2.0, 0, step=0.1, label="rep_penalty", interactive=True)
                        sum_mnts = gr.Slider(64, 2048, 0, step=1, label="new_tokens", interactive=True)
                        sum_beams = gr.Slider(1, 8, 0, step=1, label="beams", interactive=True)
                        sum_cache = gr.Radio([True, False], value=0, label="cache", interactive=True)
                        sum_sample = gr.Radio([True, False], value=0, label="sample", interactive=True)
                        sum_eosid = gr.Number(value=0, visible=False, precision=0)
                        sum_padid = gr.Number(value=0, visible=False, precision=0)

                with gr.Column(visible=False):
                    gr.Markdown("#### Context managements")
                    with gr.Row():
                        ctx_num_lconv = gr.Slider(2, 6, 3, step=1, label="num of last talks to keep", interactive=True)
                        ctx_sum_prompt = gr.Textbox(
                            "summarize our conversations. what have we discussed about so far?",
                            label="design a prompt to summarize the conversations"
                        )

        btns = [
            llama_deus_7b, koalpaca, evolinstruct_vicuna_13b, baize_7b, guanaco_33b,
        ]
        for btn in btns:
            btn.click(
                move_to_second_view,
                btn,
                [
                    model_choice_view, model_review_view,
                    model_image, model_name, model_params, model_base, model_ckpt,
                    progress_view
                ]
            )

        back_to_model_choose_btn.click(
            move_to_first_view,
            None,
            [model_choice_view, model_review_view]
        )
        
        confirm_btn.click(
            get_model_num,
            [model_name],
            [progress_view2, model_num]
        ).then(
            move_to_third_view,
            model_num,
            [progress_view2, model_review_view, chat_view, chatbot, chat_state,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid]
            # sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid]
        )
         
        for btn in channel_btns:
            btn.click(
                set_chatbot,
                [btn, local_data, chat_state],
                [chatbot, idx, example_block, regenerate]
            ).then(
                None, btn, None, 
                _js=UPDATE_LEFT_BTNS_STATE        
            )
        
        for btn in ex_btns:
            btn.click(
                set_example,
                [btn],
                [instruction_txtbox, example_block]  
            )

        instruction_txtbox.submit(
            lambda: [
                gr.update(visible=False),
                gr.update(interactive=True)
            ],
            None,
            [example_block, regenerate]
        ).then(
            central.chat_stream,
            [idx, local_data, instruction_txtbox, chat_state, model_num,
            ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid],
            [instruction_txtbox, chatbot, context_inspector, local_data],
        ).then(
            None, local_data, None, 
            _js="(v)=>{ setStorage('local_data',v) }"
        )

        regenerate.click(
            rollback_last,
            [idx, local_data, chat_state],
            [instruction_txtbox, chatbot, local_data, regenerate]
        ).then(
            central.chat_stream,
            [idx, local_data, instruction_txtbox, chat_state, model_num,
            ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid],
            [instruction_txtbox, chatbot, context_inspector, local_data],            
        ).then(
            lambda: gr.update(interactive=True),
            None,
            regenerate
        ).then(
            None, local_data, None, 
            _js="(v)=>{ setStorage('local_data',v) }"  
        )
        
        # stop.click(
        #     None, None, None,
        #     cancels=[send_event]
        # )

        clean.click(
            reset_chat,
            [idx, local_data, chat_state],
            [instruction_txtbox, chatbot, local_data, example_block, regenerate]
        ).then(
            None, local_data, None, 
            _js="(v)=>{ setStorage('local_data',v) }"
        )
        
        demo.load(
          None,
          inputs=None,
          outputs=[chatbot, local_data],
          _js=GET_LOCAL_STORAGE,
        )          
        
demo.queue(
    concurrency_count=5,
    max_size=256,
).launch(
    server_port=6006, 
    server_name="0.0.0.0", 
    debug=True
)
