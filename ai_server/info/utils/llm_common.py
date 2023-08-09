# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
from info import llm_dict


def llm_stream_generate(model_name, prompt_list, history_list, base_query_list, sources=None, **kwargs):
    for resp_list, history_list in llm_dict[model_name]['model'].lets_stream_chat(prompt_list, history_list, **kwargs):
        responses = []
        for i in range(len(resp_list)):
            history_list[i][-1][0] = base_query_list[i]
            history_list[i][-1][1] = resp_list[i]

            if sources[i] is not None:
                responses.append(
                    {'answer': resp_list[i], 'history': history_list[i][-10:],
                     'source': sources[i]})
            else:
                responses.append(
                    {'answer': resp_list[i], 'history': history_list[i][-10:]})
        yield json.dumps(responses, ensure_ascii=False)


def llm_generate(model_name, prompt_list, history_list, base_query_list, sources=None, **kwargs):
    resp_list = llm_dict[model_name]['model'].letschat(prompt_list, history_list, **kwargs)
    responses = []

    for i in range(len(base_query_list)):
        history_list[i].append([base_query_list[i], resp_list[i]])

        responses.append({'answer': resp_list[i], 'history': history_list[i][-10:], 'source': sources[i]})

    return responses
