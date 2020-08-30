#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 5/9/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterAccounting.py
# @Note    : 用于从财务报表中提取财务数据

import re
import os
from baseClass import *

#数据读写处理的基类
class InterpreterBase(BaseClass):
    def __init__(self,gConfig):
        super(InterpreterBase, self).__init__(gConfig)
        self.interpreter_name = self._get_class_name(self.gConfig)
        self.working_directory = os.path.join(self.gConfig['working_directory'],'interpreter',self.interpreter_name)
        self.logging_directory = os.path.join(self.gConfig['logging_directory'], 'interpreter', self.interpreter_name)
        self.program_directory = self.gConfig['program_directory']
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        self._get_interpreter_keyword()
        #self.interpretDefine()

    def _get_class_name(self, gConfig):
        #获取解释器的名称
        dataset_name = re.findall('Interpreter(.*)', self.__class__.__name__).pop().lower()
        assert dataset_name in gConfig['interpreterlist'], \
            'interpreterlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['interpreterlist'], dataset_name, self.__class__.__name__)
        return dataset_name

    def _get_interpreter_keyword(self):
        #编译器,文件解析器共同使用的关键字
        self.tokens = self.gJsonInterpreter['tokens']
        self.literals = self.gJsonInterpreter['literals']
        self.ignores = self.gJsonInterpreter['ignores']
        self.dictTokens = {token:value for token,value in self.gJsonInterpreter.items() if token in self.tokens}
        self.tableNames = [tableName for tableName in self.gJsonInterpreter['TABLE'].split('|')]
        self.dictTables = {keyword: value for keyword, value in self.gJsonInterpreter.items() if
                           keyword in self.tableNames}

    def _get_keyword(self,tableKeyword):
        #获取解析文件所需的关键字
        dictKeyword = {keyword:value for keyword,value in self.gJsonInterpreter.items() if keyword in tableKeyword}
        return dictKeyword

    def _get_invalid_field(self):
        return [NONESTR,NULLSTR]

    def interpretDefine(self):
        #定义一个解释器语言词法
        pass

    def initialize(self):
        #初始化一个解释器语言
        pass
