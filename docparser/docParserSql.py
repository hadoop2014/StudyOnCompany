#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# coding: utf-8
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
import numpy as np
#from pandas.io import sql
import datetime
import sys
from importlib import reload

reload(sys)

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
        table = self._merge_header(table, tableName)
        #dataframe = pd.DataFrame(table)
        dataframe = pd.DataFrame(table[1:],columns=table[0],index=None)
        isHorizontalTable = self.dictTables[tableName]['horizontalTable']
        discardField = self.dictTables[tableName]['discardField']
        discardHeader = self.dictTables[tableName]['discardHeader']
        #针对合并所有者权益表的前三列空表头进行合并
        #dataframe = self._merge_header(dataframe,tableName)
        #去掉空字段
        indexDiscardField = dataframe.iloc[:,0].isin(discardField)
        countDiscardField = indexDiscardField.sum()
        dataframe.loc[indexDiscardField] = np.nan
        dataframe = dataframe.dropna(axis=0)
        if not isHorizontalTable:
            dataframe = dataframe.T.copy()
        #去掉无用的表头
        dataframe.loc[dataframe.index.isin(discardHeader)] = np.nan
        dataframe = dataframe.dropna(axis=0)
        # dataframe前面插入公共字段
        self._add_common_field(dataframe, dictTable, countDiscardField)

        for i in range(1,len(dataframe.index)):
            sql_df = pd.DataFrame(dataframe.iloc[i]).T
            sql_df.columns = dataframe.iloc[0].values
            self._write_to_sqlite3(tableName, sql_df)

    def _write_to_sqlite3(self, tableName, dataFrame):
        conn = self._get_connect()
        isRecordExist = self._is_record_exist(conn, tableName, dataFrame)
        if not isRecordExist:
            dataFrame.to_sql(name=tableName,con=conn,if_exists='append',index=None)
            conn.commit()
        conn.close()

    def _merge_header(self,table,tableName):
        #针对合并所有者权益表的前三空表头进行合并
        #去掉第一行
        #dataFrame.iloc[0] = np.nan
        #dataFrame = dataFrame.dropna(axis=0).copy()
        tableInter = table[1:]
        #tableInter = table
        fieldFromHeader = self.dictTables[tableName]["fieldFromHeader"]
        mergedHeader = None

        index = 0
        if fieldFromHeader != "":
            for row in tableInter:
                if row[0] != '':
                    break
                else:
                    if index == 0:
                        mergedHeader = row #tolist把data
                    else:
                        mergedHeader = [self.select(field1,field2) for field1,field2 in zip(mergedHeader,row)]
                    #dataFrame.iloc[index] = np.nan
                index += 1

            if mergedHeader is not None :
                #pandas的columns不支持中文字符
                #mergedHeader = [field.replace('：','') for field in mergedHeader]
                #dataFrame.columns = mergedHeader
                tableInter[index-1] = mergedHeader
                tableInter = tableInter[index-1:]
                #dataFrame = dataFrame.dropna(axis=0)
        if index == 0:
            tableInter = table
        return tableInter
    '''
    def _merge_header(self,dataFrame,tableName):
        #针对合并所有者权益表的前三空表头进行合并
        #去掉第一行
        dataFrame.iloc[0] = np.nan
        dataFrame = dataFrame.dropna(axis=0).copy()
        fieldFromHeader = self.dictTables[tableName]["fieldFromHeader"]
        mergedHeader = None

        if fieldFromHeader != "":
            for index,field in enumerate(dataFrame.iloc[:,0]):
                if field != '':
                    break
                else:
                    if index == 0:
                        mergedHeader = dataFrame.iloc[index].tolist() #tolist把data
                    else:
                        mergedHeader = [self.select(field1,field2) for field1,field2 in zip(mergedHeader,dataFrame.iloc[index].tolist())]
                    dataFrame.iloc[index] = np.nan
            if mergedHeader is not None :
                #pandas的columns不支持中文字符
                mergedHeader = [field.replace('：','') for field in mergedHeader]
                dataFrame.columns = mergedHeader
                dataFrame = dataFrame.dropna(axis=0)
        return dataFrame
        '''
    def select(self,x1,x2):
            if x2 != '':
                return x2
            else:
                return x1

    def _add_common_field(self, dataFrame, dictTable, countDiscardField):
        #在dataFrame前面插入公共字段
        tableName = dictTable["tableName"]
        fieldFromHeader = self.dictTables[tableName]["fieldFromHeader"]
        countColumns = len(dataFrame.columns) + countDiscardField
        index = 0
        for (commonFiled, _) in self.commonFileds.items():
            if commonFiled == "ID":
                #跳过ID字段,该字段为数据库自增字段
                continue
            if commonFiled == "报告时间":
                #公共字段为报告时间时,需要特殊处理
                if fieldFromHeader != "":
                    #针对分季度财务指标,指标都是同一年的,但是分了四个季度
                    value =  [commonFiled,*[str(int(dictTable[commonFiled].split('年')[0])) + '年'
                                       for i in range(len(dataFrame.iloc[:,0])-1)]]
                else:
                    value = [commonFiled,*[str(int(dictTable[commonFiled].split('年')[0]) - i) + '年'
                                       for i in range(len(dataFrame.iloc[:,0])-1)]]
            else:
                value = [commonFiled,*[dictTable[commonFiled]]*(len(dataFrame.iloc[:,0])-1)]
            dataFrame.insert(index,column=countColumns,value=value)
            countColumns += 1
            index += 1
        #在公共字段后插入由表头转换来的字段
        if fieldFromHeader != "":
            value = dataFrame.index.values
            value[0] = fieldFromHeader
            dataFrame.insert(index,column=countColumns,value=value)

        return dataFrame

    def _is_record_exist(self, conn, tableName, dataFrame):
        fieldFromHeader = self.dictTables[tableName]["fieldFromHeader"]
        primaryKey = [key for key,value in self.commonFileds.items() if value.find('NOT NULL') >= 0]
        condition = ' and '.join([str(key) + '=\"' + str(dataFrame[key].values[0]) + '\"' for key in primaryKey])
        sql = 'select count(*) from {} where '.format(tableName) + condition
        if fieldFromHeader != "":
            #对于分季度财务数据,报告时间都是同一年,所以必须通过季度来判断记录是否唯一
            sql = sql + ' and {} = \"{}\"'.format(fieldFromHeader,dataFrame[fieldFromHeader].values[0])
        result = conn.execute(sql).fetchall()
        isRecordExist = False
        if len(result) > 0:
            isRecordExist = (result[0][0] > 0)
        return isRecordExist

    def _get_connect(self):
        return sqlite.connect(self.database)

    def _get_engine(self):
        return create_engine(os.path.join('sqlite:///',self.database))

    def _is_table_exist(self, cursor, tableName):
        isTableExist = True
        sql = '''select count(*) from sqlite_master where type = 'table' and name = %s'''%tableName
        result = cursor.execute(sql)
        if result.getInt(0) == 0 :
            isTableExist = True
        return isTableExist

    def _fetch_all_tables(self, cursor):
        try:
            cursor.execute("select name from sqlite_master where type='table' order by name")
        except Exception as e:
            print(e)
        return cursor.fetchall()

    def _create_tables(self):
        conn = self._get_connect()
        cursor = conn.cursor()
        allTables = self._fetch_all_tables(cursor)
        allTables = list(map(lambda x:x[0],allTables))
        for tableName in self.tablesNames:
            if tableName not in allTables:
                sql = ''' CREATE TABLE IF NOT EXISTS [%s] ( \n\t\t\t\t\t''' % tableName
                # if self._isTableExist(cursor,tableName) == False:
                for commonFiled, type in self.commonFileds.items():
                    sql = sql + "[%s] %s\n\t\t\t\t\t," % (commonFiled, type)
                #由表头转换生产的字段
                fieldFromHeader = self.dictTables[tableName]["fieldFromHeader"]
                if fieldFromHeader != "":
                    sql = sql + "[%s] CHAR(20)\n\t\t\t\t\t,"%fieldFromHeader
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