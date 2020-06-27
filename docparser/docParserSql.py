#!/usr/bin/env Python
# coding=utf-8
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

        dataframe = pd.DataFrame(table[1:],columns=table[0],index=None)

        #针对合并所有者权益表的前三列空表头进行合并
        dataframe,countHeaderMerge = self._header_merge(dataframe,tableName)

        #去掉空字段
        dataframe,countFieldDiscard = self._process_field_discard(dataframe, tableName)

        #去掉无用的表头;同时对水平表进行转置,把字段名由index转为column
        dataframe = self._process_header_discard(dataframe, tableName)

        #把表字段名进行标准化,标准化之后再去重复
        dataframe,countFieldStandardize = self._process_field_standardize(dataframe,tableName)

        #同一张表的相同字段在不同财务报表中名字不同,需要统一为相同名称,统一后再去重
        dataframe = self._process_field_alias(dataframe,tableName)

        #处理重复字段
        dataframe = self._process_field_duplicate(dataframe,tableName)

        #dataframe前面插入公共字段
        dataframe = self._add_common_field(dataframe, dictTable, countFieldDiscard + countHeaderMerge + countFieldStandardize)

        #把dataframe写入sqlite3数据库
        self._write_to_sqlite3(tableName, dataframe)

    def _write_to_sqlite3(self, tableName, dataFrame):
        conn = self._get_connect()
        for i in range(1,len(dataFrame.index)):
            sql_df = pd.DataFrame(dataFrame.iloc[i]).T
            sql_df.columns = dataFrame.iloc[0].values
            isRecordExist = self._is_record_exist(conn, tableName, sql_df)
            if not isRecordExist:
                sql_df.to_sql(name=tableName,con=conn,if_exists='append',index=None)
                conn.commit()
        conn.close()

    def _header_merge(self,dataFrame,tableName):
        #针对合并所有者权益表的前三空表头进行合并
        fieldFromHeader = self.dictTables[tableName]["fieldFromHeader"]
        mergedHeader = None
        countMergeHeader = 0

        def select_header(field1, field2):
            if field2 != '':
                return field2
            else:
                return field1

        if fieldFromHeader != "":
            for index,field in enumerate(dataFrame.iloc[:,0]):
                if field != '':
                    break
                else:
                    if index == 0:
                        mergedHeader = dataFrame.iloc[index].tolist()
                    else:
                        mergedHeader = [select_header(field1,field2) for field1,field2
                                        in zip(mergedHeader,dataFrame.iloc[index].tolist())]
                    dataFrame.iloc[index] = np.nan
                    countMergeHeader += 1
            if mergedHeader is not None :
                dataFrame.columns = mergedHeader
                dataFrame = dataFrame.dropna(axis=0)
        return dataFrame,countMergeHeader

    def _add_common_field(self, dataFrame, dictTable, countFieldDiscard):
        #在dataFrame前面插入公共字段
        tableName = dictTable["tableName"]
        fieldFromHeader = self.dictTables[tableName]["fieldFromHeader"]
        countColumns = len(dataFrame.columns) + countFieldDiscard
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
        dataFrame = self._process_field_from_header(dataFrame,fieldFromHeader,index,countColumns)
        return dataFrame


    def _process_field_from_header(self,dataFrame,fieldFromHeader,index,countColumns):
        #在公共字段后插入由表头转换来的字段
        if fieldFromHeader != "":
            value = dataFrame.index.values
            value[0] = fieldFromHeader
            dataFrame.insert(index,column=countColumns,value=value)
        return dataFrame

    def _process_field_duplicate(self,dataFrame,tableName):
        # 重复字段处理,放在字段标准化之后
        fieldDuplicate = self.dictTables[tableName]['fieldDuplicate']
        dictFieldDuplicate = {}
        def duplicate(fieldName):
            if fieldName in fieldDuplicate:
                dictFieldDuplicate.update({fieldName: dictFieldDuplicate[fieldName] + 1})
                if dictFieldDuplicate[fieldName] > 1:
                    fieldName = fieldName + str(dictFieldDuplicate[fieldName] - 1)
            return fieldName

        if fieldDuplicate != "":
            dictFieldDuplicate = dict(zip(fieldDuplicate, [0] * len(fieldDuplicate)))
            fields = dataFrame.iloc[0].apply(duplicate)
            dataFrame.iloc[0] = fields
        return dataFrame

    def _process_header_discard(self, dataFrame, tableName):
        #去掉无用的表头;同时对水平表进行转置,把字段名由index转为column
        isHorizontalTable = self.dictTables[tableName]['horizontalTable']
        headerDiscard = self.dictTables[tableName]['headerDiscard']
        if not isHorizontalTable:
            #同时对水平表进行转置,把字段名由index转为column,便于插入sqlite3数据库
            dataFrame = dataFrame.T.copy()
            dataFrame.loc[dataFrame.index.isin(headerDiscard)] = np.nan
            dataFrame = dataFrame.dropna(axis=0).copy()
        return dataFrame

    def _process_field_discard(self, dataFrame, tableName):
        #去掉空字段,针对主要会计数据这张表,需要提出掉其空字段
        #对于普通股现金分红情况表,则忽略这一过程
        fieldDiscard = self.dictTables[tableName]['fieldDiscard']
        indexDiscardField = dataFrame.iloc[:,0].isin(fieldDiscard)
        countDiscardField = indexDiscardField.sum()
        dataFrame.loc[indexDiscardField] = np.nan
        dataFrame = dataFrame.dropna(axis=0).copy()
        return dataFrame,countDiscardField

    def _process_field_alias(self,dataFrame,tableName):
        #同一张表的相同字段在不同财务报表中名字不同,需要统一为相同名称
        fieldAlias = self.dictTables[tableName]['fieldAlias']
        fieldAliasKeys = list(self.dictTables[tableName]['fieldAlias'].keys())

        def alias(field):
            if field in fieldAliasKeys:
                field = fieldAlias[field]
            return field

        if len(fieldAliasKeys) > 0:
            fields = dataFrame.iloc[0].apply(alias)
            dataFrame.iloc[0] = fields
        return dataFrame

    def _process_field_standardize(self,dataFrame,tableName):
        #把表字段进行标准化,把所有的字段名提取为两种模式,如:利息收入,一、营业总收入
        fieldStandardize = self.dictTables[tableName]['fieldStandardize']
        countFieldStandardize = 0
        def standardize(field):
            matched = re.search(fieldStandardize,field)
            if matched is not None:
                return  matched[0]
            else:
                return  np.nan
        if fieldStandardize != "":
            fields = dataFrame.iloc[0].apply(standardize)
            dataFrame.iloc[0] = fields
            countFieldStandardize = fields.isna().sum()
            dataFrame = dataFrame.dropna(axis = 1).copy()

        return dataFrame,countFieldStandardize

    def _is_record_exist(self, conn, tableName, dataFrame):
        #用于数据在插入数据库之前,通过组合的关键字段判断记录是否存在.
        fieldFromHeader = self.dictTables[tableName]["fieldFromHeader"]
        primaryKey = [key for key,value in self.commonFileds.items() if value.find('NOT NULL') >= 0]
        #对于Sqlit3,字符串表示为'string' ,而不是"string".
        condition = ' and '.join([str(key) + '=\'' + str(dataFrame[key].values[0]) + '\'' for key in primaryKey])
        sql = 'select count(*) from {} where '.format(tableName) + condition
        if fieldFromHeader != "":
            #对于分季度财务数据,报告时间都是同一年,所以必须通过季度来判断记录是否唯一
            sql = sql + ' and {} = \'{}\''.format(fieldFromHeader,dataFrame[fieldFromHeader].values[0])
        result = conn.execute(sql).fetchall()
        isRecordExist = False
        if len(result) > 0:
            isRecordExist = (result[0][0] > 0)
        return isRecordExist

    def _get_connect(self):
        #用于获取数据库连接
        return sqlite.connect(self.database)

    def _get_engine(self):
        return create_engine(os.path.join('sqlite:///',self.database))

    '''
    def _is_table_exist(self, cursor, tableName):
        isTableExist = True
        sql = 'select count(*) from sqlite_master where type = 'table' and name = %s'%tableName
        result = cursor.execute(sql)
        if result.getInt(0) == 0 :
            isTableExist = True
        return isTableExist
    '''

    def _fetch_all_tables(self, cursor):
        #获取数据库中所有的表,用于判断待新建的表是否已经存在
        try:
            cursor.execute("select name from sqlite_master where type='table' order by name")
        except Exception as e:
            print(e)
        return cursor.fetchall()

    def _create_tables(self):
        #用于向Sqlite3数据库中创建新表
        conn = self._get_connect()
        cursor = conn.cursor()
        allTables = self._fetch_all_tables(cursor)
        allTables = list(map(lambda x:x[0],allTables))
        for tableName in self.tablesNames:
            if tableName not in allTables:
                sql = " CREATE TABLE IF NOT EXISTS [%s] ( \n\t\t\t\t\t" % tableName
                # if self._isTableExist(cursor,tableName) == False:
                for commonFiled, type in self.commonFileds.items():
                    sql = sql + "[%s] %s\n\t\t\t\t\t," % (commonFiled, type)
                #由表头转换生产的字段
                fieldFromHeader = self.dictTables[tableName]["fieldFromHeader"]
                if fieldFromHeader != "":
                    sql = sql + "[%s] VARCHAR(20)\n\t\t\t\t\t,"%fieldFromHeader
                sql = sql[:-1]  # 去掉最后一个逗号
                #创建新表
                fieldStandardize = self.dictTables[tableName]['fieldStandardize']
                fieldDuplicate = self.dictTables[tableName]['fieldDuplicate']
                dictFieldDuplicate = {}
                if fieldDuplicate != "":
                    dictFieldDuplicate = dict(zip(fieldDuplicate,[0]*len(fieldDuplicate)))
                for fieldName in self.dictTables[tableName]['fieldName']:
                    if fieldStandardize == "":
                        sql = sql + "\n\t\t\t\t\t,[%s]  NUMERIC"%fieldName
                    else:
                        #对表字段进行标准化,去掉不必要的字符
                        matched = re.search(fieldStandardize,fieldName)
                        if matched is not None:
                            fieldName = matched[0]
                        else:
                            fieldName = ""

                        # 重复字段处理,放在字段标准化之后
                        if fieldDuplicate != "":
                            if fieldName in fieldDuplicate:
                                dictFieldDuplicate.update({fieldName: dictFieldDuplicate[fieldName] + 1})
                                if dictFieldDuplicate[fieldName] > 1:
                                    fieldName = fieldName + str(dictFieldDuplicate[fieldName] - 1)

                        #合并利率表中有:（9）其他,模式匹配后为空,需要剔除掉
                        if fieldName != "":
                            sql = sql + "\n\t\t\t\t\t,[%s]  NUMERIC"%fieldName
                sql = sql + '\n\t\t\t\t\t)'
                try:
                    conn.execute(sql)
                    conn.commit()
                    print('创建数据库表%s成功' % (tableName))
                except Exception as e:
                    # 回滚
                    conn.rollback()
                    print(e,' 创建数据库表%s失败' % tableName)

                #创建索引
                sql = "CREATE INDEX IF NOT EXISTS [%s索引] on [%s] (\n\t\t\t\t\t"%(tableName,tableName)
                sql = sql + ", ".join(str(field) for field,value in self.commonFileds.items()
                                     if value.find('NOT NULL') >= 0)
                sql = sql + '\n\t\t\t\t\t)'
                try:
                    conn.execute(sql)
                    conn.commit()
                    print('创建数据库%s索引成功' % (tableName))
                except Exception as e:
                    # 回滚
                    conn.rollback()
                    print(e,' 创建数据库%s索引失败' % tableName)
        cursor.close()
        conn.close()

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