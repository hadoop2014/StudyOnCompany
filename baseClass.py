#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/25/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserBaseClass.py
from loggerClass import *
import functools
import numpy as np
#数据读写处理的基类

NULLSTR = ''
NONESTR = 'None'
NaN = np.nan

class BaseClass():
    def __init__(self,gConfig):
        self.gConfig = gConfig
        self.gJsonInterpreter = gConfig['gJsonInterpreter'.lower()]
        self.gJsonBase = gConfig['gJsonBase'.lower()]
        self.debugIsOn = gConfig['debugIsOn'.lower()]
        self.program_directory = gConfig['program_directory']
        self.unitestIsOn = gConfig['unittestIsOn'.lower()]
        self._data = list()
        self._index = 0
        self._length = len(self._data)
        #不同的类继承BaseClass时,logger采用不同的名字
        self.EOF = gConfig['EOF'.lower()]
        self._logger = Logger(gConfig,self._get_class_name(gConfig)).logger

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = self._data[self._index]
        except IndexError:
            raise StopIteration
        self._index += 1
        return data

    def __getitem__(self, item):
        return self._data[item]

    def _set_dataset(self,index=None):
        if isinstance(index,list):
            self._data = [page for i, page in enumerate(self._data) if i in index]
            self._index = 0
            self._length = len(self._data)

    def _get_text(self,page):
        return page

    def _merge_table(self, dictTable=None,interpretPrefix=None):
        if dictTable is None:
            dictTable = list()
        return dictTable

    def _write_table(self,tableName,table):
        pass

    def _close(self):
        pass

    def _debug_info(self):
        pass

    def _get_class_name(self,*args):
        return 'Base'

    @property
    def index(self):
        return self._index - 1

    @property
    def logger(self):
        return self._logger

    def loginfo(text = NULLSTR):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self,*args, **kwargs):
                result = func(self,*args, **kwargs)
                self._logger.info('%s %s() %s:\n\t%s' % (text, func.__name__, list([*args]), result))
                return result
            return wrapper
        return decorator
