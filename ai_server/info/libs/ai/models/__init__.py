# *_*coding:utf-8 *_*
# @Author : YueMengRui
from .chatglm_llm import ChatGLM
from .baichuan import BaiChuan


def build_model(model_type, model_name_or_path, **kwargs):
    if model_type == 'ChatGLM':
        model = ChatGLM(model_name_or_path, **kwargs)
    elif model_type == 'Baichuan':
        model = BaiChuan(model_name_or_path, **kwargs)
    else:
        raise 'not support model:{}'.format(model_type)

    return model
