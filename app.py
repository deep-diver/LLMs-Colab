import time
import json
import re
from os import listdir
from os.path import isfile, join
import gradio as gr
import args
import global_vars
from chats import central
from transformers import AutoModelForCausalLM
from miscs.styles import MODEL_SELECTION_CSS
from miscs.js import GET_LOCAL_STORAGE, UPDATE_LEFT_BTNS_STATE
from utils import get_chat_interface, get_chat_manager, get_global_context

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

    guard_vram = 5 * 1024.
    vram_req_full = int(info["vram(full)"]) + guard_vram
    vram_req_8bit = int(info["vram(8bit)"]) + guard_vram
    vram_req_4bit = int(info["vram(4bit)"]) + guard_vram
    
    load_mode_list = []
    
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        info["thumb"],
        f"## {btn}",
        f"**Parameters**\n: Approx. {info['parameters']}",
        f"**🤗 Hub(base)**\n: {info['hub(base)']}",
        f"**🤗 Hub(LoRA)**\n: {info['hub(ckpt)']}",
        info['desc'],
        f"""**Min VRAM requirements** :
|             half precision            |             load_in_8bit           |              load_in_4bit          | 
| ------------------------------------- | ---------------------------------- | ---------------------------------- | 
|   {round(vram_req_full/1024., 1)}GiB  | {round(vram_req_8bit/1024., 1)}GiB | {round(vram_req_4bit/1024., 1)}GiB |
""",
        info['default_gen_config'],
        info['example1'],
        info['example2'],
        info['example3'],
        info['example4'],
        "",
    )


def move_to_first_view():
    return (
        gr.update(visible=True), 
        gr.update(visible=False),
        ""
    )


def get_model_num(
    model_name
):
    model_num = 0
    re_tag = re.compile(r'<[^>]+>')
    model_name = re_tag.sub('', model_name).strip()
    print(model_name)    
    
    for idx, item in enumerate(global_vars.models):
        if item["model_name"] == model_name:
            model_num = idx
            print(idx)
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
        get_global_context(global_vars.models[model_num]["model_type"]),
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
      
        with gr.Row(elem_id="container"):
            with gr.Column():
                gr.Markdown("### NOTE")
                gr.Markdown(":If you want to re-select a model after entering the chat mode, you can do it by refreshing the web page. Sorry for the inconvenience for now, but it will be replaced with a better UI/UX soon")                
                with gr.Row(elem_classes=["sub-container"]):
                    # with gr.Column(min_width=20):
                    #     llama_deus_7b = gr.Button("llama-deus-7b", elem_id="llama-deus-7b", elem_classes=["square"])
                    #     gr.Markdown("LLaMA Deus", elem_classes=["center"])                    

                    with gr.Column(min_width=20):                        
                        baize_7b = gr.Button("baize-7b", elem_id="baize-7b", elem_classes=["square"])
                        gr.Markdown("Baize", elem_classes=["center"])                            
                        
#                     with gr.Column(min_width=20):
#                         koalpaca = gr.Button("koalpaca", elem_id="koalpaca", elem_classes=["square"])
#                         gr.Markdown("koalpaca", elem_classes=["center"])                        
                        
                    with gr.Column(min_width=20):
                        evolinstruct_vicuna_13b = gr.Button("evolinstruct-vicuna-13b", elem_id="evolinstruct-vicuna-13b", elem_classes=["square"])
                        gr.Markdown("EvolInstruct Vicuna", elem_classes=["center"])                      
                        
                    with gr.Column(min_width=20):
                        guanaco_13b = gr.Button("guanaco-13b", elem_id="guanaco-13b", elem_classes=["square"])
                        gr.Markdown("Guanaco", elem_classes=["center"])
                        
                    with gr.Column(min_width=20):
                        nous_hermes_13b = gr.Button("nous-hermes-13b", elem_id="nous-hermes-13b", elem_classes=["square"])
                        gr.Markdown("Nous Hermes", elem_classes=["center"])                        
                        
                progress_view = gr.Textbox(label="Progress")

    with gr.Column(visible=False) as model_review_view:
        gr.Markdown("# Confirm the chosen model", elem_classes=["center"])

        with gr.Column(elem_id="container2"):
            gr.Markdown("Please expect loading time to be longer than expected. Depending on the size of models, it will probably take from 100 to 300 seconds or so. Especially, expect the longest loading time with MPT model.")

            with gr.Row():
                model_image = gr.Image(None, interactive=False, show_label=False)
                with gr.Column():
                    model_name = gr.Markdown("**Model name**")
                    model_desc = gr.Markdown("...")                        
                    model_params = gr.Markdown("Parameters\n: ...")             
                    model_base = gr.Markdown("🤗 Hub(base)\n: ...")
                    model_ckpt = gr.Markdown("🤗 Hub(LoRA)\n: ...")
                    model_vram = gr.Markdown(f"""**Minimal VRAM requirement** :
|          half precision        |        load_in_8bit       |         load_in_4bit      | 
| ------------------------------ | ------------------------- | ------------------------- | 
|   {round(7830/1024., 1)}GiB    | {round(5224/1024., 1)}GiB | {round(4324/1024., 1)}GiB |
""")
                    model_thumbnail_tiny = gr.Textbox("", visible=False)

            with gr.Column():
                gen_config_path = gr.Dropdown(
                    response_configs,
                    value=response_configs[0],
                    interactive=False,
                    label="Gen Config(response)",
                )

                with gr.Accordion("Example showcases", open=False):
                    with gr.Tab("Ex1"):
                        example_showcase1 = gr.Chatbot(
                            [("hello", "world"), ("damn", "good")]
                        )
                    with gr.Tab("Ex2"):
                        example_showcase2 = gr.Chatbot(
                            [("hello", "world"), ("damn", "good")]
                        )
                    with gr.Tab("Ex3"):
                        example_showcase3 = gr.Chatbot(
                            [("hello", "world"), ("damn", "good")]
                        )
                    with gr.Tab("Ex4"):
                        example_showcase4 = gr.Chatbot(
                            [("hello", "world"), ("damn", "good")]
                        )

            with gr.Row():
                back_to_model_choose_btn = gr.Button("Back")
                confirm_btn = gr.Button("Confirm")

            with gr.Column(elem_classes=["progress-view"]):
                txt_view = gr.Textbox(label="Status")
                progress_view2 = gr.Textbox(label="Progress")
                
    with gr.Column(visible=False) as chat_view:
        idx = gr.State(0)
        model_num = gr.State(0)
        chat_state = gr.State()
        local_data = gr.JSON({}, visible=False)
        
        with gr.Row():
            with gr.Column(scale=1, min_width=180):
                gr.Markdown("GradioChat", elem_id="left-top")

                with gr.Column(elem_id="left-pane"):
                    chat_back_btn = gr.Button("Back", elem_id="chat-back-btn")
                    
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
                        regenerate = gr.Button("Rege", interactive=False, elem_classes=["aux-btn"])
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

        with gr.Accordion("Control Panel", open=False) as control_panel:
            with gr.Column():
                with gr.Column():
                    gr.Markdown("#### Global context")
                    with gr.Accordion("global context will persist during conversation, and it is placed at the top of the prompt", open=False):
                        global_context = gr.Textbox(
                            "global context",
                            lines=5,
                            max_lines=10,
                            interactive=True,
                            elem_id="global-context"
                        )

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

                with gr.Column():
                    gr.Markdown("#### Context managements")
                    with gr.Row():
                        ctx_num_lconv = gr.Slider(2, 10, 3, step=1, label="number of recent talks to keep", interactive=True)
                        ctx_sum_prompt = gr.Textbox(
                            "summarize our conversations. what have we discussed about so far?",
                            label="design a prompt to summarize the conversations",
                            visible=False
                        )
        btns = [
            baize_7b, nous_hermes_13b, evolinstruct_vicuna_13b, guanaco_13b
            # baize_7b, evolinstruct_vicuna_13b, guanaco_13b, nous_hermes_13b
            # llama_deus_7b, koalpaca, evolinstruct_vicuna_13b, baize_7b, guanaco_33b,
        ]
        for btn in btns:
            btn.click(
                move_to_second_view,
                btn,
                [
                    model_choice_view, model_review_view,
                    model_image, model_name, model_params, model_base, model_ckpt,
                    model_desc, model_vram, gen_config_path,
                    example_showcase1, example_showcase2, example_showcase3, example_showcase4,
                    progress_view
                ]
            )

        back_to_model_choose_btn.click(
            move_to_first_view,
            None,
            [model_choice_view, model_review_view, progress_view2]
        )
        
        confirm_btn.click(
            get_model_num,
            [model_name],
            [progress_view2, model_num]
        ).then(
            move_to_third_view,
            model_num,
            [progress_view2, model_review_view, chat_view, chatbot, chat_state, global_context,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid]
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
            global_context, ctx_num_lconv, ctx_sum_prompt,
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
            global_context, ctx_num_lconv, ctx_sum_prompt,
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
        
        chat_back_btn.click(
            lambda: [gr.update(visible=False), gr.update(visible=True)],
            None,
            [chat_view, model_choice_view]
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
