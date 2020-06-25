#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 6/13/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpretUnifiedSql.py
# @Note    : 用统一的SQL语言处理excel表格,操作数据库等

from interpreter.interpretBaseClass import *

class interpretUnifiedSql(interpretBase):
    def __init__(self,gConfig,docParser):
        super(interpretUnifiedSql,self).__init__(gConfig)
        self.docParser = docParser
        self.interpretDefine()