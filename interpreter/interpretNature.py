#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 6/9/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpretNature.py
# @Note    : 用接近自然语言的解释器处理各类事务,用于处理财务数据爬取,财务数据提取,财务数据分析.

from interpreter.interpretBaseClass import *


class InterpretNature(InterpretBase):
    def __init__(self,gConfig,docParser):
        super(InterpretNature, self).__init__(gConfig)
        self.docParser = docParser
        self.interpretDefine()