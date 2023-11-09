# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
from . import sqa_blu
from info import limiter, llm_dict
from flask import request, jsonify, current_app, Response
from info.utils.response_code import RET, error_map
from info.utils.llm_common import llm_stream_generate, llm_generate
from info.utils.task_data_handler import task_data_handler


@sqa_blu.route('/ai/llm/list', methods=['GET'])
@limiter.limit("120 per minute", override_defaults=False)
def support_llm_list():
    return jsonify(errcode=RET.OK, errmsg=error_map[RET.OK], data={'llm_list': list(llm_dict.keys())})


@sqa_blu.route('/ai/llm/chat', methods=['POST'])
@limiter.limit("15 per minute", override_defaults=False)
def llm_chat_sqa():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    queries = json_data.get('queries', [])
    model_name = json_data.get('model_name', None)

    current_app.logger.info(str({'model_name': model_name, 'queries': queries}) + '\n')

    if not queries:
        return jsonify(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR])

    model_name_list = list(llm_dict.keys())
    if model_name is None or model_name not in model_name_list:
        model_name = model_name_list[0]

    origin_query_list, prompt_list, history_list, sources, generation_configs, custom_configs = task_data_handler.auto_handler(
        queries)

    responses = llm_generate(model_name, prompt_list, history_list, origin_query_list, sources, **generation_configs,
                             **custom_configs)
    return jsonify(errcode=RET.OK, errmsg=error_map[RET.OK], data={'answers': responses})


@sqa_blu.route('/ai/llm/stream/chat', methods=['POST'])
@limiter.limit("15 per minute", override_defaults=False)
def llm_chat_sqa_stream():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    queries = json_data.get('queries', [])
    model_name = json_data.get('model_name', None)

    current_app.logger.info(str({'model_name': model_name, 'queries': queries}) + '\n')

    if not queries:
        return jsonify(errcode=RET.PROMPT, errmsg=error_map[RET.PROMPT])

    model_name_list = list(llm_dict.keys())
    if model_name is None or model_name not in model_name_list:
        model_name = model_name_list[0]

    resp_code,origin_query_list, prompt_list, history_list, sources, generation_configs, custom_configs = task_data_handler.auto_handler(
        queries,model_name)

    if resp_code == 404:
        return jsonify(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR])

    return Response(llm_stream_generate(model_name, prompt_list, history_list, origin_query_list, sources, **generation_configs,
                                        **custom_configs), mimetype='text/event-stream')
