# -*- coding: utf-8 -*-
# @Time    : 12/9/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserSql.py
# @Note    : 用于sql数据库的读写

from docparser.docParserBaseClass import  *
import sqlite3 as sqlite
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
from pandas.io import sql
import datetime

Base = declarative_base()

#深度学习模型的基类
class docParserSql(docParserBase):
    def __init__(self,gConfig):
        super(docParserSql,self).__init__(gConfig)
        self.database = os.path.join(gConfig['working_directory'],gConfig['database'])
        self._create_tables()

    def writeToStore(self, dictTable):
        table = dictTable['table']
        tableName = dictTable['tableName']
        dataframe = pd.DataFrame(table[1:],columns=table[0],index=None)
        isHorizontalTable = self.dictTables[tableName]['horizontalTable']
        # dataframe前面插入公共字段
        if not isHorizontalTable:
            dataframe = dataframe.T
        #header = pd.DataFrame(dataframe.loc[0]).T
        countColumns = len(dataframe.columns)
        for index, (commonFiled, _) in enumerate(self.commonFileds.items()):
            # dataframe.loc[index] = [commonFiled,*[dictTable[commonFiled]]*(len(savedTable[0])-1)]
            #if index == 0:
            #    dataframe.loc[0][0] = commonFiled
            #else:
            dataframe.insert(index,column=countColumns,value=[commonFiled,*[dictTable[commonFiled]]*(len(table[0])-1)])
            countColumns += 1
        conn = sqlite.connect(self.database)
        #dataframe = dataframe.T
        for index,year in enumerate(dataframe[0]):
            try:
                year = datetime.datetime.strptime(year.split('年')[0],'%Y').date()
                if isinstance(year, datetime.date):
                    sql.write_frame(dataframe.loc[index], name=tableName, con=conn, if_exists='append')
            except Exception as e:
                pass
        conn.commit()
        conn.close()

    def _isTableExist(self,cursor,tableName):
        isTableExist = True
        sql = '''select count(*) from sqlite_master where type = 'table' and name = %s'''%tableName
        result = cursor.execute(sql)
        if result.getInt(0) == 0 :
            isTableExist = True
        return isTableExist

    def _fetchAllTables(self,cursor):
        try:
            cursor.execute("select name from sqlite_master where type='table' order by name")
        except Exception as e:
            print(e)
        return cursor.fetchall()

    def _create_tables(self):
        conn = sqlite.connect(self.database)
        cursor = conn.cursor()
        allTables = self._fetchAllTables(cursor)
        allTables = list(map(lambda x:x[0],allTables))
        for tableName in self.tablesNames:
            if tableName not in allTables:
                sql = ''' CREATE TABLE IF NOT EXISTS [%s] ( \n\t\t\t\t\t''' % tableName
                # if self._isTableExist(cursor,tableName) == False:
                for commonFiled, type in self.commonFileds.items():
                    sql = sql + "[%s] %s\n\t\t\t\t\t," % (commonFiled, type)
                sql = sql[:-1]  # 去掉最后一个逗号
                #创建新表
                for filedName in self.dictTables[tableName]['fieldName']:
                    sql = sql + '''\n\t\t\t\t\t,[%s]  NUMERIC'''%filedName
                sql = sql + '\n\t\t\t\t\t)'
                try:
                    conn.execute(sql)
                    conn.commit()
                    print('创建数据库表%s成功' % (tableName))
                except Exception as e:
                    # 回滚
                    conn.rollback()
                    print(e,' 创建数据库表%s失败' % tableName)
        cursor.close()
        conn.close()

    def _write_table(self,tableName,dataframe):
        pass

    def initialize(self):
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        self.clear_logging_directory(self.logging_directory)

def create_object(gConfig):
    parser = docParserSql(gConfig)
    parser.initialize()
    return parser