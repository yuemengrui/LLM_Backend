# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import shutil
from flask import current_app
import datetime


def get_base_temp_files_dir(d='llm'):
    temp_file_dir = os.path.join(current_app.config['TEMP_FILE_DIR'], datetime.datetime.now().strftime('%Y%m%d'), d)
    if not os.path.exists(temp_file_dir):
        os.makedirs(temp_file_dir)

    return temp_file_dir


def response_filter(response):
    return response


def remove_temp(file_path):
    try:
        os.remove(file_path)
    except Exception as e:
        current_app.logger.error(str({'EXCEPTION': e}) + '\n')


def have_chinese(context):
    for s in context:
        if '\u4e00' <= s <= '\u9fa5':
            return True

    return False
