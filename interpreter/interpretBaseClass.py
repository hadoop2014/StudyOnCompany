#!/usr/bin/env Python
# coding   : utf-8

import re
import os
from ply import lex,yacc
from baseClass import *

#数据读写处理的基类
class InterpretBase(BaseClass):
    def __init__(self,gConfig):
        super(InterpretBase, self).__init__(gConfig)
        self.gConfig = gConfig
        self.gConfigJson = gConfig['gConfigJson']
        self.interpreter_name = self.get_interpreter_name(self.gConfig)
        self.working_directory = os.path.join(self.gConfig['working_directory'],'interpreter',self.interpreter_name)
        self.logging_directory = os.path.join(self.gConfig['logging_directory'], 'interpreter', self.interpreter_name)
        self.mainprogram = self.gConfig['mainprogram']
        self.unitestIsOn = self.gConfig['unittestIsOn'.lower()]
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        #self.get_interpreter_keyword()
        self.interpretDefine()

    def get_interpreter_name(self,gConfig):
        #获取解释器的名称
        dataset_name = re.findall('Interpret(.*)', self.__class__.__name__).pop().lower()
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
