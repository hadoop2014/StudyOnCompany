#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/25/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserBaseClass.py
from loggerClass import *
import functools
import re
import numpy as np
#数据读写处理的基类

NULLSTR = ''
NONESTR = 'None'
NaN = np.nan

class BaseClass():
    def __init__(self,gConfig):
        self.gJsonAccounting = gConfig['gJsonAccounting']
        self.gJsonBase = gConfig['gJsonBase']
        self._get_interpreter_keyword()
        self._data = list()
        self._index = 0
        self._length = len(self._data)
        #不同的类继承BaseClass时,logger采用不同的名字
        #self.NONE = gConfig['NONE'.lower()]
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

    def _get_interpreter_keyword(self):
        #编译器,文件解析器共同使用的关键字
        self.tokens = self.gJsonAccounting['tokens']
        self.literals = self.gJsonAccounting['literals']
        self.ignores = self.gJsonAccounting['ignores']
        self.criticals = self.gJsonAccounting['CRITICAL'].split('|')
        self.criticalAlias = self.gJsonAccounting['criticalAlias']
        self.dictTokens = {token:value for token,value in self.gJsonAccounting.items() if token in self.tokens}
        self.tableAlias = self.gJsonAccounting['tableAlias']
        #tableNames标准化,去掉正则表达式中的$^
        self.tableNames = [self._standardize("[\\u4E00-\\u9FA5]{3,}",tableName) for tableName in self.gJsonAccounting['TABLE'].split('|')]
        self.tableNames = list(set([self._get_tablename_alias(tableName) for tableName in self.tableNames]))
        self.dictTables = {keyword: value for keyword,value in self.gJsonAccounting.items() if keyword in self.tableNames}
        self.commonFileds = self.gJsonAccounting['公共表字段定义']
        self.tableKeyword = self.gJsonAccounting['TABLE']
        self.dictKeyword = self._get_keyword(self.tableKeyword)


    def _get_critical_alias(self,critical):
        aliasedCritical = self._alias(critical, self.criticalAlias)
        return aliasedCritical

    def _get_tablename_alias(self,tablename):
        aliasedTablename = self._alias(tablename, self.tableAlias)
        return aliasedTablename

    def _alias(self, name, dictAlias):
        alias = name
        aliasKeys = dictAlias.keys()

        if len(aliasKeys) > 0:
            if name in aliasKeys:
                alias = dictAlias[name]
        return alias

    def _merge(self,field1, field2,isFieldJoin=True):
        if self._is_valid(field2):
            if self._is_valid(field1):
                if isFieldJoin == True:
                    return field1 + field2
                else:
                    return field2
            else:
                return field2
        else:
            return field1

    def _standardize(self,fieldStandardize,field):
        standardizedField = field
        if isinstance(field, str) and isinstance(fieldStandardize, str) and fieldStandardize !="":
            matched = re.search(fieldStandardize, field)
            if matched is not None:
                standardizedField = matched[0]
            else:
                standardizedField = NaN
        else:
            if not self._is_valid(field):
                standardizedField = NaN
        return standardizedField

    def _is_valid(self, field):
        isFieldValid = False
        if isinstance(field,str):
            if field not in self._get_invalid_field():
                isFieldValid = True
        return isFieldValid

    def _get_keyword(self,tableKeyword):
        #获取解析文件所需的关键字
        dictKeyword = {keyword:value for keyword,value in self.gJsonAccounting.items() if keyword in tableKeyword}
        return dictKeyword

    def _get_invalid_field(self):
        return [NONESTR,NULLSTR]

    def _set_dataset(self,index=None):
        if isinstance(index,list):
            self._data = [page for i, page in enumerate(self._data) if i in index]
            self._index = 0
            self._length = len(self._data)

    def _get_text(self,page):
        return page

    #def _get_tables(self,data=None):
    #    if data is None:
    #        data = list()
    #    return data

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
    def loginfo(text = NULLSTR):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self,*args, **kwargs):
                result = func(self,*args, **kwargs)
                self._logger.info('%s %s() %s:\n\t%s' % (text, func.__name__, list([*args]), result))
                return result
            return wrapper
        return decorator
