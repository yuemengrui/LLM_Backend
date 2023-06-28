# *_*coding:utf-8 *_*
# @Author : YueMengRui
from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def letschat(self, **kwargs):
        """
        return answer history
        """

    @abstractmethod
    def lets_stream_chat(self, **kwargs):
        """
        yield answer history
        """
