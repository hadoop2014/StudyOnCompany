# -*- coding: utf-8 -*-
# @Time    : 12/9/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserSql.py
# @Note    : 用于sql数据库的读写

from docparser.docParserBaseClass import  *
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

#深度学习模型的基类
class docBaseSql(docParserBase):
    def __init__(self,gConfig):
        super(docBaseSql,self).__init__(gConfig)

