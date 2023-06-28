# *_*coding:utf-8 *_*
# @Author : YueMengRui
from flask import Blueprint

knowledge_blu = Blueprint('Knowledge', __name__)

from . import views
