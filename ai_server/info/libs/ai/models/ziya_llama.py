# *_*coding:utf-8 *_*
# @Author : YueMengRui
from transformers import LlamaTokenizer, LlamaForCausalLM
from .base_model import BaseModel
import torch


def torch_gc(device):
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class ZIYALlama(BaseModel):

    def __init__(self, model_name_or_path, device='cuda', **kwargs):
        self.device = torch.device(device)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        self.model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")

    def letschat(self, prompt, history=[], max_length=2048, top_p=0.7, temperature=0.95, history_len=6):

        history = history[-history_len:]

        query = '<human>:' + prompt.strip() + '\n<bot>:'
        if history:
            inp = ''
            for i in history:
                inp += '<human>:' + i[0] + '\n<bot>:' + i[1] + '\n'

            inputs = inp + query
        else:
            inputs = query

        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids.to(self.device)
        generate_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_length,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=1.,
            eos_token_id=2,
            bos_token_id=1,
            pad_token_id=0)
        outputs = generate_ids.tolist()[0][len(input_ids[0]):]
        output = self.tokenizer.decode(outputs)
        out = output.strip('<s>').strip('</s>')

        history.append([prompt.strip(), out])
        # torch_gc(self.device)

        return out, history

    def lets_stream_chat(self, **kwargs):
        pass
