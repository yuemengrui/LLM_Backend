# *_*coding:utf-8 *_*
from config import *
from flask import Flask, request
from flask_cors import CORS
from flask_limiter import Limiter
from logging.handlers import TimedRotatingFileHandler

__all__ = ['app', 'limiter', 'llm', 'knowledge_vector_store']


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

llm = build_model(model_type=LLM_MODEL_TYPE,
                  model_name_or_path=LLM_MODEL_NAME_OR_PATH,
                  history_len=LLM_HISTORY_LEN,
                  logger=app.logger)

knowledge_vector_store = KnowledgeVectorStore(vector_store_root_dir=VS_ROOT_DIR,
                                              embedding_model_name_or_path=EMBEDDING_MODEL_NAME_OR_PATH,
                                              prompt_template=KNOWLEDGE_PROMPT_TEMPLATE,
                                              embedding_device=EMBEDDING_DEVICE,
                                              vector_search_top_k=VECTOR_SEARCH_TOP_K,
                                              chunk_size=CHUNK_SIZE,
                                              score_threshold=SCORE_THRESHOLD,
                                              first_rate=FIRST_RATE,
                                              logger=app.logger)


from info.modules.SQA import sqa_blu
app.register_blueprint(sqa_blu)
from info.modules.Knowledge import knowledge_blu
app.register_blueprint(knowledge_blu)
# from info.modules.MindMap import mindmap_blu
# app.register_blueprint(mindmap_blu)
# from info.modules.Summary import summary_blu
# app.register_blueprint(summary_blu)
# from info.modules.Translation import translation_blu
# app.register_blueprint(translation_blu)
# from info.modules.Write import write_blu
# app.register_blueprint(write_blu)
