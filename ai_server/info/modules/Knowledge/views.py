# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import json
import requests
import datetime
from . import knowledge_blu
from info import llm, knowledge_vector_store, limiter
from flask import request, jsonify, current_app
from info.utils.response_code import RET, error_map
from info.utils.MD5_Utils import check_md5
from info.utils.common import response_filter, get_base_temp_files_dir, remove_temp


@knowledge_blu.route('/ai/llm/knowledge/file/add', methods=['POST'])
@limiter.limit("60 per minute", override_defaults=False)
def llm_knowledge_file_add():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    file_url = json_data.get('file_url')
    file_hash = json_data.get('file_hash')

    if not all([file_hash, file_url]):
        return jsonify(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR])

    if knowledge_vector_store.check_vector_exist(file_hash):
        return jsonify(errcode=RET.OK, errmsg=error_map[RET.OK])

    try:
        if '134.175.246' in file_url:
            file_search_path = file_url.split('ai/file')[-1][1:]
            file_path = os.path.join(current_app.config['TEMP_FILE_DIR'], file_search_path)
        else:
            file_ext = '.' + file_url.split('.')[-1]
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            file_path = os.path.join(get_base_temp_files_dir(), str(nowtime) + file_hash + file_ext)
            file_data = requests.get(file_url).content
            with open(file_path, 'wb') as f:
                f.write(file_data)
    except Exception as e:
        current_app.logger.error(e)
        return jsonify(errcode=RET.DATAERR, errmsg=error_map[RET.DATAERR])

    if not check_md5(file_path, file_hash):
        remove_temp(file_path)
        return jsonify(errcode=RET.DATAERR, errmsg=error_map[RET.DATAERR])

    if knowledge_vector_store.build_vector_store(file_path, file_hash):
        remove_temp(file_path)
        return jsonify(errcode=RET.OK, errmsg=error_map[RET.OK])
    else:
        remove_temp(file_path)
        return jsonify(errcode=RET.DATAERR, errmsg=error_map[RET.DATAERR])


@knowledge_blu.route('/ai/llm/knowledge/reinit', methods=['GET'])
@limiter.limit("60 per minute", override_defaults=False)
def llm_knowledge_reinit():
    knowledge_vector_store.init_knowledge()

    return jsonify(errcode=RET.OK, errmsg=error_map[RET.OK])
