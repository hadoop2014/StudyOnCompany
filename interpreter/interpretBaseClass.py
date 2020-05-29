import re
import os
import ply.lex as lex
import ply.yacc as yacc
from baseClass import *

#数据读写处理的基类
class interpretBase(baseClass):
    def __init__(self,gConfig):
        super(interpretBase,self).__init__()
        self.gConfig = gConfig
        self.gConfigJson = gConfig['gConfigJson']
        self.interpreter_name = self.get_interpreter_name(self.gConfig)
        self.working_directory = os.path.join(self.gConfig['working_directory'],'interpreter',self.interpreter_name)
        self.logging_directory = os.path.join(self.gConfig['logging_directory'], 'interpreter', self.interpreter_name)
        self.unitestIsOn = self.gConfig['unittestIsOn'.lower()]
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        self.get_interpreter_keyword()
        self.interpretDefine()

    def get_interpreter_keyword(self):
        self.tokens = self.gConfigJson['tokens']
        self.literals = self.gConfigJson['literals']
        self.ignores = self.gConfigJson['ignores']
        self.dictTokens = {token:value for token,value in self.gConfigJson.items() if token in self.tokens}
        self.tables = self.gConfigJson['TABLE'].split('|')
        self.dictTables = {keyword: value for keyword,value in self.gConfigJson.items() if keyword in self.tables}

    def get_interpreter_name(self,gConfig):
        #获取解释器的名称
        dataset_name = re.findall('interpret(.*)', self.__class__.__name__).pop().lower()
        assert dataset_name in gConfig['interpreterlist'], \
            'interpreterlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['datasetlist'], dataset_name, self.__class__.__name__)
        return dataset_name

    def interpretDefine(self):
        #定义一个解释器语言词法
        pass


    def initialize(self):
        #初始化一个解释器语言
        pass
