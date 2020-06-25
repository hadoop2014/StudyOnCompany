#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 6/13/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpretAnalysis.py
# @Note    : 用于财务数据分析

from interpreter.interpretBaseClass import *

class interpretAnalysis(interpretBase):
    def __init__(self,gConfig,docParser):
        super(interpretAnalysis,self).__init__(gConfig)
        self.docParser = docParser
        #self.initialize()
        self.interpretDefine()
