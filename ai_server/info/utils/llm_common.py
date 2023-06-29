# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
from info import llm
from flask import current_app


def llm_stream_generate(prompt_list, history_list, max_prompt_length, base_query_list, sources=None, **kwargs):
    for resp_list, history_list in llm.lets_stream_chat(prompt_list, history_list, max_prompt_length,
                                                        **kwargs):
        responses = []
        for i in range(len(resp_list)):
            history_list[i][-1][0] = base_query_list[i]
            history_list[i][-1][1] = resp_list[i]

            if sources[i] is not None:
                responses.append(
                    {'answer': resp_list[i], 'history': history_list[i][-current_app.config['LLM_HISTORY_LEN']:],
                     'source': sources[i]})
            else:
                responses.append(
                    {'answer': resp_list[i], 'history': history_list[i][-current_app.config['LLM_HISTORY_LEN']:]})
        yield json.dumps(responses, ensure_ascii=False)
