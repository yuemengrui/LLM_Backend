# *_*coding:utf-8 *_*
# @Author : YueMengRui
from flask import Blueprint

embedding_blu = Blueprint('Embedding', __name__)

from . import views
