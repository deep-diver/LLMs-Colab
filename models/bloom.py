import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.bettertransformer import BetterTransformer

def load_model(
    base, 
    finetuned, 
    mode_cpu,
    mode_mps,
    mode_full_gpu,
    mode_8bit,
    mode_4bit,
    force_download_ckpt
):
    tokenizer = AutoTokenizer.from_pretrained(base)

    if mode_cpu:
        print("cpu mode")
        model = AutoModelForCausalLM.from_pretrained(
            base, 
            device_map={"": "cpu"}, 
            use_safetensors=False
        )
        
        if finetuned is not None and \
            finetuned != "" and \
            finetuned != "N/A":

            model = PeftModel.from_pretrained(
                model, 
                finetuned,
                device_map={"": "cpu"}
                # force_download=force_download_ckpt,
            )
        else:
            model = BetterTransformer.transform(model)
            
    elif mode_mps:
        print("mps mode")
        model = AutoModelForCausalLM.from_pretrained(
            base,
            device_map={"": "mps"},
            torch_dtype=torch.float16,
            use_safetensors=False
        )
        
        if finetuned is not None and \
            finetuned != "" and \
            finetuned != "N/A":

            model = PeftModel.from_pretrained(
                model, 
                finetuned,
                torch_dtype=torch.float16,
                device_map={"": "mps"}
                # force_download=force_download_ckpt,
            )
        else:
            model = BetterTransformer.transform(model)
            
    else:
        print("gpu mode")
        print(f"8bit = {mode_8bit}, 4bit = {mode_4bit}")
        model = AutoModelForCausalLM.from_pretrained(
            base,
            load_in_8bit=mode_8bit,
            load_in_4bit=mode_4bit,
            device_map="auto",
            use_safetensors=False
        )

        if not mode_8bit and not mode_4bit:
            model.half()

        if finetuned is not None and \
            finetuned != "" and \
            finetuned != "N/A":

            model = PeftModel.from_pretrained(
                model, 
                finetuned, 
                # force_download=force_download_ckpt,
        )
        else:
            model = BetterTransformer.transform(model)
            
    return model, tokenizer