# *_*coding:utf-8 *_*
import os
import logging
from config import config_dict
from flask import Flask
from copy import deepcopy
from flask_cors import CORS
from flask_limiter import Limiter
from sentence_transformers import SentenceTransformer
from logging.handlers import TimedRotatingFileHandler

import nltk
nltk.data.path = ['./nltk_data'] + nltk.data.path

def setup_logging(log_level):
    logging.basicConfig(level=log_level)
    # file_log_handler = RotatingFileHandler("logs/log", maxBytes=1024 * 1024 * 100, backupCount=10)
    file_log_handler = TimedRotatingFileHandler("logs/log", when="MIDNIGHT", backupCount=30)
    file_log_handler.suffix = "%Y-%m-%d.log"
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d %(message)s')
    file_log_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_log_handler)


def api_limit_key_func():
    return '127.0.0.1'
    # return request.remote_addr or "127.0.0.1"


limiter = Limiter(
    key_func=api_limit_key_func,
    # 默认方案 每小时2000，每分钟100，适用于所有路线。如果想忽略此全局配置，方法上增加此注解@limiter.exempt
    default_limits=["60 per minute"]
)

app = Flask(__name__, static_folder='', static_url_path='')
CORS(app)
config_cls = config_dict['dev']
app.config.from_object(config_cls)
if not os.path.exists(app.config['TEMP_FILE_DIR']):
    os.makedirs(app.config['TEMP_FILE_DIR'])
if not os.path.exists(app.config['VS_ROOT_DIR']):
    os.makedirs(app.config['VS_ROOT_DIR'])

app.json.ensure_ascii = False

if app.config['LOGGER_MODE'] == 'gunicorn':
    gunicorn_logger = logging.getLogger('gunicorn.access')
    app.logger = gunicorn_logger
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
else:
    setup_logging(config_cls.LOG_LEVEL)

limiter.init_app(app)

from info.libs.ai import build_model, KnowledgeVectorStore

llm = build_model(model_type=app.config['LLM_MODEL_TYPE'],
                  model_name_or_path=app.config['LLM_MODEL_NAME_OR_PATH'],
                  history_len=app.config['LLM_HISTORY_LEN'],
                  device=app.config['LLM_DEVICE'],
                  logger=app.logger)

knowledge_vector_store = KnowledgeVectorStore(vector_store_root_dir=app.config['VS_ROOT_DIR'],
                                              embedding_model_name_or_path=app.config['EMBEDDING_MODEL_LIST'][0][
                                                  'model_path'],
                                              embedding_device=app.config['EMBEDDING_MODEL_LIST'][0]['device'],
                                              logger=app.logger)

# embedding_model_list = []
#
# for i, d in enumerate(deepcopy(app.config['EMBEDDING_MODEL_LIST'])):
#     if i == 0:
#         del d['model_path']
#         del d['device']
#         d.update({"embedding_model": knowledge_vector_store.embeddings.client})
#         embedding_model_list.append(d)
#     else:
#         model_path = d.pop('model_path')
#         device = d.pop('device')
#
#         if model_path:
#             embedding_model = SentenceTransformer(model_name_or_path=model_path, device=device)
#
#             d.update({"embedding_model": embedding_model})
#             embedding_model_list.append(d)


from info.modules.SQA import sqa_blu
app.register_blueprint(sqa_blu)
from info.modules.Knowledge import knowledge_blu
app.register_blueprint(knowledge_blu)
# from info.modules.Embedding import embedding_blu
# app.register_blueprint(embedding_blu)
