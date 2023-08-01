# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
from . import embedding_blu
from copy import deepcopy
from info import limiter, embedding_model_list
from flask import request, jsonify, current_app
from info.utils.response_code import RET, error_map
from info.libs.ai.local_knowledge_handlers.chinese_text_splitter import ChineseTextSplitter


@embedding_blu.route('/ai/text/embedding', methods=['POST'])
@limiter.limit("60 per minute", override_defaults=False)
def text_embedding():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    sentences = json_data.get('sentences', [])
    text_split = json_data.get('text_split', 0)

    current_app.logger.info(str({'sentences': sentences}) + '\n')

    if not sentences:
        return jsonify(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR])

    res = []

    for i in embedding_model_list:
        temp = {}
        try:
            if text_split == 0:
                embeddings = i['embedding_model'].encode(sentences)
                embeddings = [x.tolist() for x in embeddings]
                temp.update({"sentences": sentences})
                temp.update({"embeddings": embeddings})
                temp.update({k: v for k, v in i.items() if k != 'embedding_model'})
                res.append(deepcopy(temp))
            else:
                text_splitter = ChineseTextSplitter()
                text = "\n".join(sentences)
                sentences = text_splitter.split_text(text)
                embeddings = i['embedding_model'].encode(sentences)
                embeddings = [x.tolist() for x in embeddings]
                temp.update({"sentences": sentences})
                temp.update({"embeddings": embeddings})
                temp.update({k: v for k, v in i.items() if k != 'embedding_model'})
                res.append(deepcopy(temp))
        except Exception as e:
            current_app.logger.error(str({'EXCEPTION': e}) + '\n')

    return jsonify(errcode=RET.OK, errmsg=error_map[RET.OK], data=res)
