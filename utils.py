from chats import alpaca
from chats import alpaca_gpt4
from chats import stablelm
from chats import koalpaca
from chats import os_stablelm
from chats import vicuna
from chats import flan_alpaca
from chats import starchat
from chats import redpajama
from chats import mpt
from chats import alpacoom
from chats import baize
from chats import guanaco

from pingpong.gradio import GradioAlpacaChatPPManager
from pingpong.gradio import GradioKoAlpacaChatPPManager
from pingpong.gradio import GradioStableLMChatPPManager
from pingpong.gradio import GradioFlanAlpacaChatPPManager
from pingpong.gradio import GradioOSStableLMChatPPManager
from pingpong.gradio import GradioVicunaChatPPManager
from pingpong.gradio import GradioStableVicunaChatPPManager
from pingpong.gradio import GradioStarChatPPManager
from pingpong.gradio import GradioMPTChatPPManager
from pingpong.gradio import GradioRedPajamaChatPPManager
from pingpong.gradio import GradioBaizeChatPPManager

from pingpong.pingpong import PPManager
from pingpong.pingpong import PromptFmt
from pingpong.pingpong import UIFmt
from pingpong.gradio import GradioChatUIFmt

class GuanacoPromptFmt(PromptFmt):
    @classmethod
    def ctx(cls, context):
        if context is None or context == "":
            return ""
        else:
            return f"""{context}
"""
        
    @classmethod
    def prompt(cls, pingpong, truncate_size):
        ping = pingpong.ping[:truncate_size]
        pong = "" if pingpong.pong is None else pingpong.pong[:truncate_size]
        return f"""### Human: {ping}
### Assistant: {pong}
"""
  
class GuanacoChatPPManager(PPManager):
    def build_prompts(self, from_idx: int=0, to_idx: int=-1, fmt: PromptFmt=GuanacoPromptFmt, truncate_size: int=None):
        if to_idx == -1 or to_idx >= len(self.pingpongs):
            to_idx = len(self.pingpongs)
            
        results = fmt.ctx(self.ctx)
        
        for idx, pingpong in enumerate(self.pingpongs[from_idx:to_idx]):
            results += fmt.prompt(pingpong, truncate_size=truncate_size)
            
        return results

class GradioGuanacoChatPPManager(GuanacoChatPPManager):
    def build_uis(self, from_idx: int=0, to_idx: int=-1, fmt: UIFmt=GradioChatUIFmt):
        if to_idx == -1 or to_idx >= len(self.pingpongs):
            to_idx = len(self.pingpongs)
        
        results = []
        
        for pingpong in self.pingpongs[from_idx:to_idx]:
            results.append(fmt.ui(pingpong))
            
        return results    
    
def get_chat_interface(model_type):
    if model_type == "alpaca":
        return alpaca.chat_stream
    elif model_type == "alpaca-gpt4":
        return alpaca.chat_stream
    elif model_type == "stablelm":
        return stablelm.chat_stream
    elif model_type == "os-stablelm":
        return os_stablelm.chat_stream
    elif model_type == "koalpaca-polyglot":
        return koalpaca.chat_stream
    elif model_type == "flan-alpaca":
        return flan_alpaca.chat_stream
    elif model_type == "camel":
        return alpaca.chat_stream
    elif model_type == "t5-vicuna":
        return vicuna.chat_stream
    elif model_type == "stable-vicuna":
        return vicuna.chat_stream
    elif model_type == "starchat":
        return starchat.chat_stream
    elif model_type == "mpt":
        return mpt.chat_stream
    elif model_type == "redpajama":
        return redpajama.chat_stream
    elif model_type == "vicuna":
        return vicuna.chat_stream
    elif model_type == "llama-deus":
        return alpaca.chat_stream
    elif model_type == "evolinstruct-vicuna":
        return vicuna.chat_stream
    elif model_type == "alpacoom":
        return alpacoom.chat_stream
    elif model_type == "baize":
        return baize.chat_stream
    elif model_type == "guanaco":
        return guanaco.chat_stream
    else:
        return None

def get_chat_manager(model_type):
    if model_type == "alpaca":
        return GradioAlpacaChatPPManager
    elif model_type == "alpaca-gpt4":
        return GradioAlpacaChatPPManager()
    elif model_type == "stablelm":
        return GradioStableLMChatPPManager()
    elif model_type == "os-stablelm":
        return GradioOSStableLMChatPPManager()
    elif model_type == "koalpaca-polyglot":
        return GradioKoAlpacaChatPPManager()
    elif model_type == "flan-alpaca":
        return GradioFlanAlpacaChatPPManager()
    elif model_type == "camel":
        return GradioAlpacaChatPPManager()
    elif model_type == "t5-vicuna":
        return GradioVicunaChatPPManager()
    elif model_type == "vicuna":
        return GradioVicunaChatPPManager()
    elif model_type == "stable-vicuna":
        return GradioStableVicunaChatPPManager()
    elif model_type == "starchat":
        return GradioStarChatPPManager()
    elif model_type == "mpt":
        return GradioMPTChatPPManager()
    elif model_type == "redpajama":
        return GradioRedPajamaChatPPManager()
    elif model_type == "llama-deus":
        return GradioAlpacaChatPPManager()
    elif model_type == "evolinstruct-vicuna":
        return GradioVicunaChatPPManager()
    elif model_type == "alpacoom":
        return GradioAlpacaChatPPManager()
    elif model_type == "baize":
        return GradioBaizeChatPPManager()
    elif model_type == "guanaco":
        return GradioGuanacoChatPPManager()
    else:
        return None