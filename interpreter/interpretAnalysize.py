#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 6/13/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpretAnalysize.py
# @Note    : 用于财务数据分析

from interpreter.interpretBaseClass import *

class InterpretAnalysize(InterpretBase):
    def __init__(self,gConfig,docParser):
        super(InterpretAnalysize, self).__init__(gConfig)
        self.docParser = docParser
        #self.initialize()
        self.interpretDefine()
