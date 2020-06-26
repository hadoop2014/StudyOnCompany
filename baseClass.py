#!/usr/bin/env Python
# coding=utf-8
# @Time    : 9/25/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserBaseClass.py


#数据读写处理的基类
class baseClass():
    def __init__(self,gConfigJson):
        self.gConfigJson = gConfigJson
        self._get_interpreter_keyword()
        self.tableKeyword = gConfigJson['TABLE']
        self.dictKeyword = self._get_keyword(self.tableKeyword)
        self._data = list()
        self._index = 0
        self._length = len(self._data)

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
        self.tablesNames = self.gConfigJson['TABLE'].split('|')
        self.dictTables = {keyword: value for keyword,value in self.gConfigJson.items() if keyword in self.tablesNames}
        self.commonFileds = self.gConfigJson['公共表字段定义']
        #识别所有的关键字字符集

    def _get_keyword(self,tableKeyword):
        #获取解析文件所需的关键字
        dictKeyword = {keyword:value for keyword,value in self.gConfigJson.items() if keyword in tableKeyword}
        #self.fieldKeyword = self.gConfig['fieldKeyword'.lower()]
        #self.excludeKeyword = self.gConfig['excludeKeyword'.lower()]
        #if len(self.excludeKeyword) == 1 and self.excludeKeyword[0] == '':
        #    self.excludeKeyword = list()  # 置空
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

