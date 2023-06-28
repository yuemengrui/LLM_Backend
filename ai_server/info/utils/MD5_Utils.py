# *_*coding:utf-8 *_*
# @Author : YueMengRui
import hashlib
from flask import current_app


def md5hex(data):
    try:
        m = hashlib.md5()
        m.update(data)
        return str(m.hexdigest())
    except Exception as e:
        current_app.logger.error(str({'EXCEPTION': e}) + '\n')
        return ''


def check_md5(file_path, md5):
    with open(file_path, 'rb') as f:
        file_data = f.read()

    file_hash = md5hex(file_data)
    if file_hash == md5:
        return True
    current_app.logger.error(str({'hash_check': '{} != {}'.format(file_hash, md5)}) + '\n')
    return False
