# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
from . import sqa_blu
from config import SQA_PROMPT_TEMPLATE, LLM_HISTORY_LEN
from info import llm, knowledge_vector_store, limiter
from flask import request, jsonify, current_app, Response
from info.utils.response_code import RET, error_map
from info.utils.common import response_filter


@sqa_blu.route('/ai/llm/chat', methods=['POST'])
@limiter.limit("15 per minute", override_defaults=False)
def llm_chat_sqa():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    queries = json_data.get('queries', [])

    current_app.logger.info(str({'queries': queries}) + '\n')

    if not queries:
        return jsonify(errcode=RET.PARAMERR, errmag=error_map[RET.PARAMERR])

    base_query_list = []
    prompt_list = []
    history_list = []
    related_docs_list = []
    for query_dict in queries:
        query = query_dict.get('query', '')
        history = query_dict.get('history', [])
        file_hashs = query_dict.get('file_hashs', [])
        generate_configs = query_dict.get('generate_configs', {})

        if len(file_hashs) == 0:
            prompt = query.strip()
            related_docs = []
        else:
            prompt, related_docs = knowledge_vector_store.generate_knowledge_based_prompt(query.strip(),
                                                                                          file_hashs,
                                                                                          max_prompt_len=
                                                                                          current_app.config[
                                                                                              'MAX_PROMPT_LENGTH'],
                                                                                          prompt_template=SQA_PROMPT_TEMPLATE)
        base_query_list.append(query)
        prompt_list.append(prompt)
        history_list.append(history)
        related_docs_list.append(related_docs)

        try:
            resp_list = llm.letschat(prompt_list, history_list, current_app.config['MAX_PROMPT_LENGTH'],
                                     **generate_configs)
            resp_list = response_filter(resp_list)
        except Exception as e:
            current_app.logger.error({'EXCEPTION': e})
            resp_list = [u'抱歉，我暂时无法回答您的问题'] * len(base_query_list)

    sources = []
    for related in related_docs_list:
        temp = {}
        for doc in related:
            file_hash = doc.metadata['file_hash']
            if file_hash in temp.keys():
                temp[file_hash]['related_content'].append(doc.page_content)
            else:
                temp[file_hash] = {'file_hash': file_hash, 'related_content': [doc.page_content]}

        source = [v for v in temp.values()]
        sources.append(source)

    responses = []
    for i in range(len(base_query_list)):
        history_list[i].append([base_query_list[i], resp_list[i]])

        responses.append({'answer': resp_list[i], 'history': history_list[i][-LLM_HISTORY_LEN:], 'source': sources[i]})

    current_app.logger.info(str({'responses': responses}) + '\n')
    return jsonify(errcode=RET.OK, errmsg=error_map[RET.OK], data={'answers': responses})


@sqa_blu.route('/ai/llm/stream/chat', methods=['POST'])
@limiter.limit("15 per minute", override_defaults=False)
def llm_chat_sqa_stream():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    queries = json_data.get('queries', [])

    current_app.logger.info(str({'queries': queries}) + '\n')

    if not queries:
        return jsonify(errcode=RET.PARAMERR, errmag=error_map[RET.PARAMERR])

    base_query_list = []
    prompt_list = []
    history_list = []
    related_docs_list = []
    for query_dict in queries:
        query = query_dict.get('query', '')
        history = query_dict.get('history', [])
        file_hashs = query_dict.get('file_hashs', [])
        generate_configs = query_dict.get('generate_configs', {})

        if len(file_hashs) == 0:
            prompt = query.strip()
            related_docs = []
        else:
            prompt, related_docs = knowledge_vector_store.generate_knowledge_based_prompt(query.strip(),
                                                                                          file_hashs,
                                                                                          max_prompt_len=
                                                                                          current_app.config[
                                                                                              'MAX_PROMPT_LENGTH'],
                                                                                          prompt_template=SQA_PROMPT_TEMPLATE)
        base_query_list.append(query)
        prompt_list.append(prompt)
        history_list.append(history)
        related_docs_list.append(related_docs)

    sources = []
    for related in related_docs_list:
        temp = {}
        for doc in related:
            file_hash = doc.metadata['file_hash']
            if file_hash in temp.keys():
                temp[file_hash]['related_content'].append(doc.page_content)
            else:
                temp[file_hash] = {'file_hash': file_hash, 'related_content': [doc.page_content]}

        source = [v for v in temp.values()]
        sources.append(source)

    current_app.logger.info(str({'sources': sources}) + '\n')

    def generate(prompt_list, history_list, max_prompt_length, **kwargs):
        for resp_list, history_list in llm.lets_stream_chat(prompt_list, history_list, max_prompt_length,
                                                            **kwargs):
            responses = []
            for i in range(len(resp_list)):
                history_list[i][-1][0] = base_query_list[i]
                history_list[i][-1][1] = resp_list[i]

                responses.append(
                    {'answer': resp_list[i], 'history': history_list[i][-LLM_HISTORY_LEN:], 'source': sources[i]})
            yield json.dumps(responses, ensure_ascii=False)

    return Response(
        generate(prompt_list, history_list, current_app.config['MAX_PROMPT_LENGTH'], **generate_configs),
        mimetype='text/event-stream')
