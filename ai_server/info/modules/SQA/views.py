# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
from . import sqa_blu
from config import SMART_QA_PROMPT_TEMPLATE, LLM_HISTORY_LEN, KNOWLEDGE_PROMPT_TEMPLATE
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
        # max_length = query_dict.get('max_length', 2048)
        # top_p = query_dict.get('top_p', 0.7)
        # temperature = query_dict.get('temperature', 0.95)
        # return_relation = query_dict.get('return_relation', 1)

        if len(file_hashs) == 0:
            prompt = query.strip()
            related_docs = []
        else:
            prompt, related_docs = knowledge_vector_store.generate_knowledge_based_prompt(query.strip(),
                                                                                          file_hashs,
                                                                                          max_prompt_len=
                                                                                          current_app.config[
                                                                                              'MAX_PROMPT_LENGTH'],
                                                                                          prompt_template=SMART_QA_PROMPT_TEMPLATE)
        base_query_list.append(query)
        prompt_list.append(prompt)
        history_list.append(history)
        related_docs_list.append(related_docs)

        # try:
        resp_list = llm.letschat(prompt_list, history_list, current_app.config['MAX_PROMPT_LENGTH'])

        resp_list = response_filter(resp_list)

    # except Exception as e:
    #     current_app.logger.error({'EXCEPTION': e})
    #     resp_list = [u'抱歉，我暂时无法回答您的问题'] * len(base_query_list)

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
    max_length_list = []
    top_p_list = []
    temperature_list = []
    for query_dict in queries:
        query = query_dict.get('query', '')
        history = query_dict.get('history', [])
        file_hashs = query_dict.get('file_hashs', [])
        max_length = query_dict.get('max_length', 2048)
        top_p = query_dict.get('top_p', 0.8)
        temperature = query_dict.get('temperature', 0.8)
        score_rate = query_dict.get('score_rate', 0.1)
        # return_relation = query_dict.get('return_relation', 1)

        if len(file_hashs) == 0:
            prompt = query.strip()
            related_docs = []
        else:
            prompt, related_docs = knowledge_vector_store.generate_knowledge_based_prompt(query.strip(),
                                                                                          file_hashs,
                                                                                          max_prompt_len=
                                                                                          current_app.config[
                                                                                              'MAX_PROMPT_LENGTH'],
                                                                                          prompt_template=KNOWLEDGE_PROMPT_TEMPLATE,
                                                                                          score_rate=score_rate)
        base_query_list.append(query)
        prompt_list.append(prompt)
        history_list.append(history)
        related_docs_list.append(related_docs)
        max_length_list.append(max_length)
        top_p_list.append(top_p)
        temperature_list.append(temperature)

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
    max_length = max(max_length_list)
    top_p = max(top_p_list)
    temperature = max(temperature_list)

    def generate(prompt_list, history_list, max_prompt_length, max_length, top_p, temperature):
        for resp_list, history_list in llm.lets_stream_chat(prompt_list, history_list, max_prompt_length,
                                                            max_length=max_length, top_p=top_p,
                                                            temperature=temperature):
            responses = []
            for i in range(len(resp_list)):
                history_list[i][-1][0] = base_query_list[i]
                history_list[i][-1][1] = resp_list[i]

                responses.append(
                    {'answer': resp_list[i], 'history': history_list[i][-LLM_HISTORY_LEN:], 'source': sources[i]})
            yield json.dumps(responses, ensure_ascii=False)

    return Response(
        generate(prompt_list, history_list, current_app.config['MAX_PROMPT_LENGTH'], max_length, top_p, temperature),
        mimetype='text/event-stream')
