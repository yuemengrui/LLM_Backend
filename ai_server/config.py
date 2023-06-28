# *_*coding:utf-8 *_*
import os
import json
import logging
from urllib import parse

LLM_MODEL_TYPE = 'ChatGLM'
LLM_MODEL_NAME_OR_PATH = ''

EMBEDDING_MODEL_NAME_OR_PATH = ''
EMBEDDING_DEVICE = "cuda:1"
# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 10
FIRST_RATE = 0.1

# LLM input history length
LLM_HISTORY_LEN = 10
CHUNK_SIZE = 512
SCORE_THRESHOLD = 150

OCR_URL = 'http://127.0.0.1:5000/ai/ocr/general'

VS_ROOT_DIR = "./vector_store"

if not os.path.exists(VS_ROOT_DIR):
    os.makedirs(VS_ROOT_DIR)

# 基于上下文的prompt模版，请务必保留"{query}"和"{context}"
KNOWLEDGE_PROMPT_TEMPLATE = """已知信息：
{context} 
请扮演一名专业分析师，根据上述已知信息专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，回答时不要说“根据已知信息”这几个字，请直接返回答案，答案请使用中文。问题是：{query}"""

SQA_PROMPT_TEMPLATE = """已知信息：
{context} 
请扮演一名专业分析师，根据上述已知信息专业的来回答用户的问题。如果无法从中得到答案，请忽略已知信息。回答时不要说“根据已知信息”这几个字，请直接返回答案, 问题是：{query}"""

TRANSLATION_PROMPT_TEMPLATE = "这是一个中英翻译的任务，将输入文本翻译成另一种语言，如果输入文本是中文则翻译成英文，如果输入文本是英文则翻译成中文。\n输入文本: {context}"
TRANSLATION2CHINESE_PROMPT_TEMPLATE = "请将输入文本翻译成中文。\n输入文本: {context}"
TRANSLATION2ENGLISH_PROMPT_TEMPLATE = "请将输入文本翻译成英文。\n输入文本: {context}"

SUMMARY_PROMPT_TEMPLATE = "这是一个文本摘要的任务，分析并总结已知内容，然后输出摘要。\n已知内容:\n {context}"
WRITE_PROMPT_TEMPLATE = "分析并总结已知内容，根据已知内容来回答问题。\n已知内容:\n {context} \n 问题：{query}"

MARKDOWN_PROMPT_TEMPLATE = """已知信息：
{context} 
分析并总结上述已知信息，写一篇思维导图，以markdown格式返回结果。"""


class Config(object):
    SECRET_KEY = 'YueMengRui-LLM'

    JSON_AS_ASCII = False

    # 默认日志等级
    LOG_LEVEL = logging.INFO
    LOGGER_MODE = 'gunicorn'

    TEMP_FILE_DIR = './temp_files'
    MAX_PROMPT_LENGTH = 3000


class DevelopmentConfig(Config):
    """开发模式下的配置"""
    DEBUG = logging.INFO


class UatConfig(Config):
    """生产模式下的配置"""
    LOG_LEVEL = logging.INFO


class ProductionConfig(Config):
    """生产模式下的配置"""
    LOG_LEVEL = logging.INFO


config_dict = {
    "dev": DevelopmentConfig,
    "uat": UatConfig,
    "prod": ProductionConfig
}
