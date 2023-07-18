# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
from . import sqa_blu
from info import limiter
from flask import request, jsonify, current_app, Response
from info.utils.response_code import RET, error_map
from info.utils.llm_common import llm_stream_generate, llm_generate
from info.utils.task_data_handler import task_data_handler


@sqa_blu.route('/ai/llm/chat', methods=['POST'])
@limiter.limit("15 per minute", override_defaults=False)
def llm_chat_sqa():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    queries = json_data.get('queries', [])

    current_app.logger.info(str({'queries': queries}) + '\n')

    if not queries:
        return jsonify(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR])

    origin_query_list, prompt_list, history_list, sources, generation_configs, custom_configs = task_data_handler.auto_handler(
        queries)

    responses = llm_generate(prompt_list, history_list, origin_query_list, sources, **generation_configs,
                             **custom_configs)
    return jsonify(errcode=RET.OK, errmsg=error_map[RET.OK], data={'answers': responses})


@sqa_blu.route('/ai/llm/stream/chat', methods=['POST'])
@limiter.limit("15 per minute", override_defaults=False)
def llm_chat_sqa_stream():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    queries = json_data.get('queries', [])

    current_app.logger.info(str({'queries': queries}) + '\n')

    if not queries:
        return jsonify(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR])

    origin_query_list, prompt_list, history_list, sources, generation_configs, custom_configs = task_data_handler.auto_handler(
        queries)

    return Response(llm_stream_generate(prompt_list, history_list, origin_query_list, sources, **generation_configs,
                                        **custom_configs), mimetype='text/event-stream')
