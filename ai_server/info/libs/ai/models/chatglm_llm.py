# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
from copy import deepcopy
from .base_model import BaseModel
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModel


def torch_gc(device):
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上

    # device_map = {'transformer.word_embeddings': 0,
    #               'transformer.final_layernorm': 0, 'lm_head': 0}

    device_map = {
        'transformer.embedding.word_embeddings': 0,
        'transformer.encoder.final_layernorm': 0,
        'transformer.output_layer': 0,
        'transformer.rotary_pos_emb': 0,
        'lm_head': 0
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        # device_map[f'transformer.layers.{i}'] = gpu_target
        device_map[f'transformer.encoder.layers.{i}'] = gpu_target
        used += 1

    return device_map


class ChatGLM(BaseModel):

    def __init__(self, model_name_or_path, history_len=10, device='cuda', logger=None, **kwargs):
        self.model = None
        self.tokenizer = None
        self.history_len = history_len
        self.device = device
        self.logger = logger
        self._load_model(model_name_or_path, device)

    def _load_model(self,
                    model_name_or_path,
                    device='cuda',
                    device_map: Optional[Dict[str, int]] = None,
                    **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

        # self.model = AutoModel.from_pretrained(model_name_or_path,trust_remote_code=True).quantize(8).cuda(0)

        if torch.cuda.is_available() and device.lower().startswith("cuda"):
            # 根据当前设备GPU数量决定是否进行多卡部署
            num_gpus = torch.cuda.device_count()
            if num_gpus < 2 and device_map is None:
                self.model = (
                    AutoModel.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=True,
                        **kwargs)
                    .half()
                    .cuda()
                )
            else:
                from accelerate import dispatch_model

                model = (
                    AutoModel.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=True,
                        **kwargs)
                    .half())
                # 可传入device_map自定义每张卡的部署情况
                if device_map is None:
                    device_map = auto_configure_device_map(num_gpus)

                self.model = dispatch_model(model, device_map=device_map)
        else:
            self.model = (
                AutoModel.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    **kwargs)
                .float()
                .to(device)
            )

        self.model = self.model.eval()

    def letschat(self, query_list, history_list, max_prompt_length, max_length=4096, top_p=0.8, temperature=0.8,
                 **kwargs):

        batch_prompt = []
        for i in range(len(query_list)):
            query = query_list[i]
            history = history_list[i]

            if history and len(query) < max_prompt_length:
                sum_len = len("[Round 1]\n\n问：{}\n\n答：".format(query))
                true_history = []
                for (old_query, old_response) in history[::-1]:
                    history_prompt_len = len("[Round 1]\n\n问：{}\n\n答：{}\n\n".format(old_query, old_response))
                    if sum_len + history_prompt_len > max_prompt_length:
                        break
                    else:
                        true_history.insert(0, (old_query, old_response))
                        sum_len += history_prompt_len
                history = deepcopy(true_history)

            prompt = ""
            for j, (old_query, old_response) in enumerate(history):
                prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(j + 1, old_query, old_response)
            prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)

            if self.logger:
                self.logger.info(str({'prompt_len': len(prompt), 'prompt': prompt}) + '\n')
                self.logger.info(str({'max_length': max_length, 'top_p': top_p, 'temperature': temperature}) + '\n')
            batch_prompt.append(prompt)

        response_list = self.model.batch_chat(
            self.tokenizer,
            batch_prompt,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature
        )

        torch_gc(self.device)

        return response_list

    def lets_stream_chat(self, query_list, history_list, max_prompt_length=3096, max_length=4096, top_p=0.8,
                         temperature=0.8, **kwargs):
        if self.logger:
            self.logger.info(str({'max_length': max_length, 'top_p': top_p, 'temperature': temperature,
                                  'max_prompt_length': max_prompt_length}) + '\n')
            self.logger.info(str(kwargs) + '\n')

        batch_prompt = []
        for i in range(len(query_list)):
            query = query_list[i]
            history = history_list[i]

            if history and len(query) < max_prompt_length:
                sum_len = len("[Round 1]\n\n问：{}\n\n答：".format(query))
                true_history = []
                for (old_query, old_response) in history[::-1]:
                    history_prompt_len = len("[Round 1]\n\n问：{}\n\n答：{}\n\n".format(old_query, old_response))
                    if sum_len + history_prompt_len > max_prompt_length:
                        break
                    else:
                        true_history.insert(0, (old_query, old_response))
                        sum_len += history_prompt_len
                history = deepcopy(true_history)

            prompt = ""
            for j, (old_query, old_response) in enumerate(history):
                prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(j + 1, old_query, old_response)
            prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)

            if self.logger:
                self.logger.info(str({'prompt_len': len(prompt), 'prompt': prompt}) + '\n')

            batch_prompt.append(prompt)

        for ind in range(len(batch_prompt)):
            history_list[ind].append(['', ''])

        for resp_list in self.model.batch_stream_chat(self.tokenizer, batch_prompt, max_length=max_length, top_p=top_p,
                                                      temperature=temperature):
            yield resp_list, history_list
