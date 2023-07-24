# *_*coding:utf-8 *_*
import os
import json
import logging
from urllib import parse

OCR_URL = 'http://127.0.0.1:5000/ai/ocr/general'


class Config(object):
    SECRET_KEY = 'YueMengRui-LLM'

    JSON_AS_ASCII = False

    # 默认日志等级
    LOG_LEVEL = logging.INFO
    LOGGER_MODE = 'gunicorn'

    TEMP_FILE_DIR = './temp_files'


class DevelopmentConfig(Config):
    """开发模式下的配置"""
    LOG_LEVEL = logging.INFO

    LLM_MODEL_TYPE = ''
    LLM_MODEL_NAME_OR_PATH = ''
    LLM_DEVICE = 'cuda'
    LLM_HISTORY_LEN = 10

    EMBEDDING_MODEL_NAME_OR_PATH = ''
    EMBEDDING_DEVICE = "cuda"

    VS_ROOT_DIR = "./vector_store"


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
