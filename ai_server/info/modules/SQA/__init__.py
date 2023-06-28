# *_*coding:utf-8 *_*
# @Author : YueMengRui
from flask import Blueprint

sqa_blu = Blueprint('SQA', __name__)

from . import views
