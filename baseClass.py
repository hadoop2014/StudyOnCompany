#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/25/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserBaseClass.py
from loggerClass import *
import functools
#数据读写处理的基类
class BaseClass():
    def __init__(self,gConfig):
        self.gConfigJson = gConfig['gConfigJson']
        self._get_interpreter_keyword()
        self._data = list()
        self._index = 0
        self._length = len(self._data)
        #不同的类继承BaseClass时,logger采用不同的名字
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

    def _get_interpreter_keyword(self):
        #编译器,文件解析器共同使用的关键字
        self.tokens = self.gConfigJson['tokens']
        self.literals = self.gConfigJson['literals']
        self.ignores = self.gConfigJson['ignores']
        self.criticals = self.gConfigJson['CRITICAL'].split('|')
        self.criticalAlias = self.gConfigJson['criticalAlias']
        self.dictTokens = {token:value for token,value in self.gConfigJson.items() if token in self.tokens}
        self.tableAlias = self.gConfigJson['tableAlias']
        self.tableNames = self.gConfigJson['TABLE'].split('|')
        self.tableNames = list(set([self._get_tablename_alias(tableName) for tableName in self.tableNames]))
        self.dictTables = {keyword: value for keyword,value in self.gConfigJson.items() if keyword in self.tableNames}
        self.commonFileds = self.gConfigJson['公共表字段定义']
        self.tableKeyword = self.gConfigJson['TABLE']
        self.dictKeyword = self._get_keyword(self.tableKeyword)
        #识别所有的关键字字符集

    def _get_critical_alias(self,critical):
        #criticalAliasKeys = self.criticalAlias()
        aliasedCritical = self._get_alias(critical,self.criticalAlias)
        return aliasedCritical

    def _get_tablename_alias(self,tablename):
        #tableAliasKeys = self.tableAlias()
        aliasedTablename = self._get_alias(tablename,self.tableAlias)
        return aliasedTablename

    def _get_alias(self,name,dictAlias):
        aliase = name
        aliasKeys = dictAlias.keys()

        if len(aliasKeys) > 0:
            if name in aliasKeys:
                aliase = dictAlias[name]
        return aliase

    def _get_keyword(self,tableKeyword):
        #获取解析文件所需的关键字
        dictKeyword = {keyword:value for keyword,value in self.gConfigJson.items() if keyword in tableKeyword}
        return dictKeyword


    def _set_dataset(self,index=None):
        if isinstance(index,list):
            self._data = [page for i, page in enumerate(self._data) if i in index]
            self._index = 0
            self._length = len(self._data)

    def _get_text(self,page):
        #page = self.__getitem__(self._index)
        return page

    def _get_tables(self,data=None):
        if data is None:
            data = list()
        return data

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

    #@staticmethod
    def loginfo(text = ''):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self,*args, **kwargs):
                result = func(self,*args, **kwargs)
                self._logger.info('%s %s() %s:\n\t%s' % (text, func.__name__, list([*args]), result))
                return result
            return wrapper
        return decorator
