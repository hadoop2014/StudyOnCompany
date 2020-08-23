#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 6/13/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterAnalysize.py
# @Note    : 用于财务数据分析

from interpreterAnalysize.interpreterBaseClass import *

class InterpreterAnalysize(InterpreterBase):
    def __init__(self,gConfig,docParser):
        super(InterpreterAnalysize, self).__init__(gConfig)
        self.docParser = docParser
        #self.initialize()
        self.interpretDefine()

def create_object(gConfig,memberModuleDict):
    interpreter=InterpreterAnalysize(gConfig, memberModuleDict)
    interpreter.initialize()
    return interpreter