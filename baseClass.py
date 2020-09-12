#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/25/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserBaseClass.py
from loggerClass import *
import functools
import numpy as np
import os
import re
import sqlite3 as sqlite
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
        self.working_directory = os.path.join(self.gConfig['working_directory'], self._get_module_path())
        #self.logging_directory = os.path.join(self.gConfig['logging_directory'], self._get_module_path())
        self.unitestIsOn = gConfig['unittestIsOn'.lower()]
        self._data = list()
        self._index = 0
        self._length = len(self._data)
        #不同的类继承BaseClass时,logger采用不同的名字
        self.EOF = gConfig['EOF'.lower()]
        self._logger = Logger(gConfig,self._get_class_name(gConfig)).logger
        self.database = os.path.join(gConfig['working_directory'],gConfig['database'])

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

    def _get_module_path(self):
        module = self.__class__.__module__
        path = os.path.join(*module.split('.'))
        return path

    def _get_connect(self):
        #用于获取数据库连接
        return sqlite.connect(self.database)

    def _sql_executer(self,sql):
        conn = self._get_connect()
        #cursor = conn.cursor()
        try:
            conn.execute(sql)
            conn.commit()
            self.logger.info('success to execute sql(脚本执行成功):\n%s' % sql)
        except Exception as e:
            # 回滚
            conn.rollback()
            self.logger.error('failed to execute sql(脚本执行失败):%s\n%s' % (str(e),sql))
        #cursor.close()
        conn.close()

    def _is_matched(self,pattern,field):
        isMatched = False
        if isinstance(field, str) and isinstance(pattern, str) and pattern != NULLSTR:
            matched = re.search(pattern, field)
            if matched is not None:
                isMatched = True
        return isMatched

    def _sql_executer_script(self,sql):
        isSuccess = False
        conn = self._get_connect()
        #cursor = conn.cursor()
        try:
            conn.executescript(sql)
            conn.commit()
            self.logger.info('success to execute sql(脚本执行成功)!')
            isSuccess = True
        except Exception as e:
            # 回滚
            conn.rollback()
            self.logger.error('failed to execute sql(脚本执行失败):%s' % (str(e)))
        #cursor.close()
        conn.close()
        return isSuccess

    def _get_file_context(self,fileName):
        file_object = open(fileName,encoding='utf-8')
        file_context = NULLSTR
        try:
            file_context = file_object.read()  # file_context是一个string，读取完后，就失去了对test.txt的文件引用
        except Exception as e:
            self.logger.error('读取文件(%s)失败:%s' % (fileName,str(e)))
        finally:
            file_object.close()
        return file_context

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
