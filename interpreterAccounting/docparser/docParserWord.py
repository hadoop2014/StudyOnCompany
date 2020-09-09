#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 12/9/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserWord.py
# @Note    : 用于word文件的读写

from interpreterAccounting.docparser.docParserBaseClass import  *

#深度学习模型的基类
class DocBaseWord(DocParserBase):
    def __init__(self,gConfig):
        super(DocBaseWord, self).__init__(gConfig)