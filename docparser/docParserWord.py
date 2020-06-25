#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 12/9/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserWord.py
# @Note    : 用于word文件的读写

from docparser.docParserBaseClass import  *

#深度学习模型的基类
class docBaseWord(docParserBase):
    def __init__(self,gConfig):
        super(docBaseWord,self).__init__(gConfig)