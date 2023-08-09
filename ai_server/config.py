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
    LOGGER_MODE = 'gunicor'

    TEMP_FILE_DIR = './temp_files'


class DevelopmentConfig(Config):
    """开发模式下的配置"""
    LOG_LEVEL = logging.INFO
    VS_ROOT_DIR = "./vector_store"

    LLM_HISTORY_LEN = 10

    LLM_MODEL_LIST = [
        {
            "model_type": "ChatGLM",
            "model_name": "ChatGLM2-6B",
            "embedding_dim": 4096,
            "model_name_or_path": "",
            "device": "cuda"
        },
        {
            "model_type": "Baichuan",
            "model_name": "Baichuan-13B",
            "embedding_dim": 5120,
            "model_name_or_path": "",
            "device": "cuda"
        }
    ]

    EMBEDDING_MODEL_LIST = [
        {
            "embedding_type": "text",
            "model_name": "m3e-base",
            "max_seq_length": 512,
            "embedding_dim": 768,
            "model_path": "",
            "device": "cuda"
        }
    ]


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
