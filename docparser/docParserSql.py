#!/usr/bin/env Python
# coding   : utf-8
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
import datetime
import sys
from importlib import reload
import time

reload(sys)

Base = declarative_base()

#深度学习模型的基类
class DocParserSql(DocParserBase):
    def __init__(self,gConfig):
        super(DocParserSql, self).__init__(gConfig)
        self.database = os.path.join(gConfig['working_directory'],gConfig['database'])
        self._create_tables()
        self.process_info = {}
        self.dataTable = {}

    def loginfo(text = 'running '):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self,*args, **kwargs):
                result = func(self,*args, **kwargs)
                resultForLog = result
                columns = 0
                if isinstance(result,tuple):
                    resultForLog = result[0].T.copy()
                    columns = result[0].iloc[0]
                self.logger.info('%s %s() \n\t%s,%s%s,\t%s:\n\t%s\n\t columns=%s'
                                  % (text, func.__name__,
                                     self.dataTable['公司名称'],self.dataTable['报告时间'],self.dataTable['报告类型'],
                                     args[-1],
                                     resultForLog,columns))
                return result
            return wrapper
        return decorator

    def writeToStore(self, dictTable):
        self.dataTable = dictTable
        table = dictTable['table']
        tableName = dictTable['tableName']

        self.process_info.update({tableName:time.time()})
        dataframe,countTotalFields = self._table_to_dataframe(table,tableName)#pd.DataFrame(table[1:],columns=table[0],index=None)

        #针对合并所有者权益表的前三列空表头进行合并,对转置表进行预转置,使得其处理和其他表一致
        dataframe = self._process_header_merge(dataframe, tableName)

        #把跨多个单元格的表字段名合并成一个
        dataframe = self._process_field_merge(dataframe,tableName)

        #去掉空字段及无用字段
        dataframe = self._process_field_discard(dataframe, tableName)

        #去掉无用的表头;同时对水平表进行转置,把字段名由index转为column
        dataframe = self._process_header_discard(dataframe, tableName)

        #把表头进行标准化
        dataframe = self._process_header_standardize(dataframe,tableName)

        #把表字段名进行标准化,标准化之后再去重复
        dataframe = self._process_field_standardize(dataframe,tableName)

        #同一张表的相同字段在不同财务报表中名字不同,需要统一为相同名称,统一后再去重
        dataframe = self._process_field_alias(dataframe,tableName)

        #处理重复字段
        dataframe = self._process_field_duplicate(dataframe,tableName)

        #dataframe前面插入公共字段
        dataframe = self._process_field_common(dataframe, dictTable, countTotalFields,tableName)

        dataframe = self._process_value_standardize(dataframe,tableName)

        #把dataframe写入sqlite3数据库
        self._write_to_sqlite3(dataframe,tableName)
        self.process_info.update({tableName:time.time() - self.process_info[tableName]})

    @loginfo()
    def _table_to_dataframe(self,table,tableName):
        horizontalTable = self.dictTables[tableName]['horizontalTable']
        if horizontalTable == True:
            #对于装置表,如普通股现金分红情况表,不需要表头
            dataFrame = pd.DataFrame(table,index=None)
            countTotalFields = len(dataFrame.columns.values)
        else:
            #dataFrame = pd.DataFrame(table[1:],columns=table[0],index=None)
            dataFrame = pd.DataFrame(table, index=None)
            countTotalFields = len(dataFrame.index.values)
        dataFrame.fillna('None',inplace=True)
        return dataFrame,countTotalFields

    def _write_to_sqlite3(self, dataFrame,tableName):
        conn = self._get_connect()
        for i in range(1,len(dataFrame.index)):
            sql_df = pd.DataFrame(dataFrame.iloc[i]).T
            sql_df.columns = dataFrame.iloc[0].values
            isRecordExist = self._is_record_exist(conn, tableName, sql_df)
            if not isRecordExist:
                sql_df.to_sql(name=tableName,con=conn,if_exists='append',index=None)
                conn.commit()
        conn.close()

    def _process_header_merge(self, dataFrame, tableName):
        #针对合并所有者权益表的前三空表头进行合并
        #针对普通股现金分红情况表进行表头合并,因为其为转置表,实际是对其字段进行了合并.在合并完后进行预转置,使得其后续处理和其他表保持一致
        isHorizontalTable = self.dictTables[tableName]['horizontalTable']
        mergedRow = None
        firstHeader = self.dictTables[tableName]['header'][0]
        lastIndex = 0
        keepAheader = False

        #需要把插入在表中间的表头合并掉
        for index,field in enumerate(dataFrame.iloc[:,0]):
            if self._is_valid(field):
                if isHorizontalTable == True:
                    if self._is_field_first(tableName,field):
                        if index > lastIndex + 1 and mergedRow is not None:
                            #if mergedRow[0] == firstHeader:
                            dataFrame.iloc[lastIndex] = mergedRow
                            dataFrame.iloc[lastIndex + 1:index] = NaN
                            keepAheader = True
                        #mergedRow = None
                        break
                else:
                    if field != firstHeader and  (dataFrame.iloc[index] != 'None').all()  :
                        if index > lastIndex + 1 and mergedRow is not None:
                            if mergedRow[0] == firstHeader and firstHeader != NULLSTR:
                                dataFrame.iloc[lastIndex] = mergedRow
                                dataFrame.iloc[lastIndex + 1:index] = NaN
                                keepAheader = True
                        mergedRow = None
                    #else:
                    #    if firstHeader == NULLSTR:
                    #        if index > lastIndex + 1 and mergedRow is not None:
                    #            dataFrame.iloc[lastIndex] = mergedRow
                    #            dataFrame.iloc[lastIndex + 1:index] = NaN
                    #    mergedRow = None

            if mergedRow is not None and (dataFrame.iloc[index] != 'None').all():
                #在启动合并后,碰到第一行非全为None的即退出
                #mergedRow = reduce(self._merge,dataFrame.iloc[index].tolist())
                #headerStandardize = self.dictTables[tableName]['headerStandardize']
                #if self._is_field_matched(headerStandardize,mergedRow) == False:
                if self._is_header_in_row(dataFrame.iloc[index].tolist(),tableName) == False:
                    if index > lastIndex + 1 and mergedRow is not None:
                        if mergedRow[0] == firstHeader and firstHeader != NULLSTR:
                            dataFrame.iloc[lastIndex] = mergedRow
                            dataFrame.iloc[lastIndex + 1:index] = NaN
                            keepAheader = True
                    mergedRow = None

            if keepAheader and firstHeader == field:#self._is_header_in_row(dataFrame.iloc[index].tolist(),tableName) == True:
                mergedRow = None

            if mergedRow is None:
                mergedRow = dataFrame.iloc[index].tolist()
                lastIndex = index
            else:
                mergedRow = self._get_merged_row(dataFrame.iloc[index].tolist(), mergedRow, isFieldJoin=True)
            #dataFrame.iloc[index] = NaN
        if isHorizontalTable == True:
            #如果是转置表,则在此处做一次转置,后续的处理就和非转置表保持一致了
            #if mergedRow is not None:
            #    dataFrame.iloc[0] = mergedRow
            #dataFrame.iloc[0] = dataFrame
            dataFrame = dataFrame.dropna(axis=0)
            #把第一列做成索引
            dataFrame.set_index(0,inplace=True)
            dataFrame = dataFrame.T.copy()
        else:
            #if mergedRow is not None :
            columns = dataFrame.iloc[0].copy()
            #dataFrame = dataFrame.dropna(axis=0)
            indexDiscardField = dataFrame.iloc[:, 0].isin([firstHeader])
            dataFrame.loc[indexDiscardField] = NaN
            dataFrame.columns = columns
            dataFrame = dataFrame.dropna(axis=0).copy()
        return dataFrame

    @loginfo()
    def _process_field_merge(self,dataFrame,tableName):
        mergedRow = None
        lastIndex = 0
        lastField = None
        countIndex = len(dataFrame.index.values)
        mergedFields = reduce(self._merge,dataFrame.iloc[:, 0].tolist())
        isStandardizeStrictMode = self._is_standardize_strict_mode(mergedFields,tableName)

        for index, field in enumerate(dataFrame.iloc[:, 0],start=0):
            #情况1: 当前field和standardizedFields匹配成功,表示新的字段开始,则处理之前的mergedRow
            #情况2: 当前field和standardizedFields匹配不成功,又有两种情况:
            #    a)上一个mergedRow已经拼出了完整的field,和standardizedFields匹配成功,此时当前row为[field,非None,非None,...]
            #    b)上一个mergedRow还没有拼出完整的field,但是仍和standardizedFields匹配成功,此时要么当前field='None'
            #      或则当前row为[field,None,None,None,...]
            #    c)2019良信电器合并所有者权益变动表,"同一控制下企业合并"分成了多行,且全是空字符,而非None
            if self._is_valid(field):
                if self._is_field_match_standardize(field, tableName) and not (dataFrame.iloc[index][1:] == 'None').all():
                    if lastField != NULLSTR:
                        #前面一个空字段所在行必定合入到下一个非空字段中
                        if index > lastIndex + 1 and mergedRow is not None:
                            # 把前期合并的行赋值到dataframe的上一行
                             dataFrame.iloc[lastIndex] = mergedRow
                             dataFrame.iloc[lastIndex + 1:index] = NaN
                        mergedRow = None
                    else:
                        #如果前面一个是空字段, 但是同一行内包含了header内容,主要会计数据会把header插入到表中间位置.
                        if self._is_header_in_row(dataFrame.iloc[index-1].tolist(),tableName) == True:
                            if index > lastIndex + 1 and mergedRow is not None:
                                # 把前期合并的行赋值到dataframe的上一行
                                dataFrame.iloc[lastIndex] = mergedRow
                                dataFrame.iloc[lastIndex + 1:index] = NaN
                            mergedRow = None

                else:
                    if mergedRow is not None:
                        mergedField = mergedRow[0]
                        if self._is_valid(mergedField):
                            if self._is_field_in_standardize_by_mode(mergedField,isStandardizeStrictMode,tableName):
                                if index > lastIndex + 1 and mergedRow is not None:
                                    dataFrame.iloc[lastIndex] = mergedRow
                                    dataFrame.iloc[lastIndex + 1:index] = NaN
                                mergedRow = None

            elif field == NULLSTR and mergedRow is not None:
                #如果field为空的情况下,下一行的field仍旧是空行,则当前行空字段行需要并入mergedRow
                aheaderField = None
                if index + 1 < countIndex:
                    aheaderField = dataFrame.iloc[index + 1,0]
                if aheaderField != NULLSTR:
                    mergedField = mergedRow[0]
                    if self._is_valid(mergedField):
                        #当前字段为'',下一个字段有效,如果合并后的字段为标准字段,则认为合并成功
                        if self._is_field_in_standardize_by_mode(mergedField,isStandardizeStrictMode,tableName):
                            if index > lastIndex + 1 and mergedRow is not None:
                                dataFrame.iloc[lastIndex] = mergedRow
                                dataFrame.iloc[lastIndex + 1:index] = NaN
                            mergedRow = None
            else:
                #针对field = 'None'或则其他非法情况,则继续合并
                pass

            if mergedRow is None:
                mergedRow = dataFrame.iloc[index].tolist()
                lastIndex = index
            else:
                mergedRow = self._get_merged_row(dataFrame.iloc[index].tolist(), mergedRow, isFieldJoin=True)
            lastField = field
        #最后一个合并行的处理
        if mergedRow is not None:
            mergedField = mergedRow[0]
            if self._is_valid(mergedField):
                if self._is_field_in_standardize_strict(mergedField, tableName):
                    if countIndex > lastIndex + 1 and mergedRow is not None:
                        dataFrame.iloc[lastIndex] = mergedRow
                        dataFrame.iloc[lastIndex + 1:countIndex] = NaN
                    mergedRow = None
        return dataFrame

    @loginfo()
    def _process_field_common(self, dataFrame, dictTable, countFieldDiscard,tableName):
        #在dataFrame前面插入公共字段
        fieldFromHeader = self.dictTables[tableName]["fieldFromHeader"]
        countColumns = len(dataFrame.columns) + countFieldDiscard
        index = 0
        for (commonFiled, _) in self.commonFileds.items():
            if commonFiled == "ID":
                #跳过ID字段,该字段为数据库自增字段
                continue
            if commonFiled == "报告时间":
                assert dictTable[commonFiled] != NULLSTR,'dictTable[%s] must not be null!'%commonFiled
                #公共字段为报告时间时,需要特殊处理
                if fieldFromHeader != "":
                    #针对分季度财务指标,指标都是同一年的,但是分了四个季度
                    value =  [commonFiled,*[str(int(dictTable[commonFiled].split('年')[0])) + '年'
                                       for i in range(len(dataFrame.iloc[:,0])-1)]]
                else:
                    firstHeader = dataFrame.index.values[0]
                    #针对普通股现金分红情况
                    if isinstance(firstHeader,str) and firstHeader == '分红年度':
                        value = [commonFiled,*dataFrame.index[1:].tolist()]
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

    def _process_value_standardize(self,dataFrame,tableName):
        #对非法值进行统一处理
        def valueStandardize(value):
            try:
                if isinstance(value,str):
                    value = value.replace('\n', NULLSTR).replace(' ', NULLSTR).replace('None',NULLSTR)
            except Exception as e:
                print(e)
            return value
        #dataFrame = dataFrame.loc[:].apply(lambda x:x.apply(valueStandardize))
        dataFrame = dataFrame.apply(lambda x: x.apply(valueStandardize))
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
        duplicatedFields = self._get_duplicated_field(dataFrame.iloc[0].tolist())
        dataFrame.iloc[0] = duplicatedFields
        return dataFrame

    def _process_header_discard(self, dataFrame, tableName):
        #去掉无用的表头;把字段名由index转为column
        headerDiscard = self.dictTables[tableName]['headerDiscard']
        headerDiscardPattern = '|'.join(headerDiscard)
        #同时对水平表进行转置,把字段名由index转为column,便于插入sqlite3数据库
        dataFrame = dataFrame.T.copy()
        #删除需要丢弃的表头,该表头由self.dictTables[tableName]['headerDiscard']定义
        indexDiscardHeader = [self._is_field_matched(headerDiscardPattern,x) for x in dataFrame.index.values]
        dataFrame.loc[indexDiscardHeader] = NaN
        dataFrame = dataFrame.dropna(axis=0).copy()
        return dataFrame

    def _process_field_discard(self, dataFrame, tableName):
        #去掉空字段,针对主要会计数据这张表,需要提出掉其空字段
        #对于普通股现金分红情况表,则忽略这一过程
        fieldDiscard = self.dictTables[tableName]['fieldDiscard']
        indexDiscardField = dataFrame.iloc[:,0].isin(fieldDiscard+self.NONE)
        dataFrame.loc[indexDiscardField] = NaN
        dataFrame = dataFrame.dropna(axis=0).copy()
        return dataFrame

    def _process_field_alias(self,dataFrame,tableName):
        #同一张表的相同字段在不同财务报表中名字不同,需要统一为相同名称
        aliasedFields = self._get_aliased_fields(dataFrame.iloc[0].tolist(), tableName)
        dataFrame.iloc[0] = aliasedFields
        return dataFrame

    def _process_header_standardize(self,dataFrame,tableName):
        #把表头字段进行标准化
        standardizedHeaders = self._get_standardized_header(dataFrame.index.tolist(),tableName)
        dataFrame.index = standardizedHeaders
        return dataFrame

    def _process_field_standardize(self,dataFrame,tableName):
        #把表字段进行标准化,把所有的字段名提取为两种模式,如:利息收入,一、营业总收入
        standardizedFields = self._get_standardized_field(dataFrame.iloc[0].tolist(),tableName)
        dataFrame.iloc[0] = standardizedFields
        dataFrame = dataFrame.dropna(axis = 1).copy()
        return dataFrame

    def _get_aliased_fields(self, fieldList, tableName):
        aliasedFields = fieldList
        fieldAlias = self.dictTables[tableName]['fieldAlias']
        fieldAliasKeys = list(self.dictTables[tableName]['fieldAlias'].keys())

        if len(fieldAliasKeys) > 0:
            aliasedFields = [self._alias(field, fieldAlias) for field in fieldList]
        return aliasedFields

    def _get_merged_row(self, sourceRow, mergeRow, isFieldJoin=False):
        #当isHorizontalTable=True时,为转置表,如普通股现金分红情况表,这个时候是对字段合并,采用字段拼接方式,其他情况采用替换方式
        mergedRow = [self._merge(field1, field2, isFieldJoin) for field1, field2 in zip(mergeRow, sourceRow)]
        return mergedRow

    def _get_duplicated_field(self,fieldList):
        dictFieldDuplicate = dict(zip(fieldList,[0]*len(fieldList)))
        def duplicate(fieldName):
            dictFieldDuplicate.update({fieldName:dictFieldDuplicate[fieldName] + 1})
            if dictFieldDuplicate[fieldName] > 1:
                fieldName += str(dictFieldDuplicate[fieldName] - 1)
            return fieldName

        duplicatedField = [duplicate(fieldName) for fieldName in fieldList]
        return duplicatedField

    def _get_standardized_header(self,headerList,tableName):
        assert headerList is not None, 'sourceRow(%s) must not be None' % headerList
        fieldStandardize = self.dictTables[tableName]['headerStandardize']
        if isinstance(headerList, list):
            standardizedFields = [self._standardize(fieldStandardize, field) for field in headerList]
        else:
            standardizedFields = self._standardize(fieldStandardize, headerList)
        return standardizedFields

    def _get_standardized_field_strict(self,fieldList,tableName):
        assert fieldList is not None, 'sourceRow(%s) must not be None' % fieldList
        fieldStandardizeStrict = self.dictTables[tableName]['fieldStandardizeStrict']
        if isinstance(fieldList, list):
            standardizedFields = [self._standardize(fieldStandardizeStrict, field) for field in fieldList]
        else:
            standardizedFields = self._standardize(fieldStandardizeStrict, fieldList)
        return standardizedFields

    def _get_standardized_field(self,fieldList,tableName):
        assert fieldList is not None,'sourceRow(%s) must not be None'%fieldList
        fieldStandardize = self.dictTables[tableName]['fieldStandardize']
        if isinstance(fieldList,list):
            standardizedFields = [self._standardize(fieldStandardize,field) for field in fieldList]
        else:
            standardizedFields = self._standardize(fieldStandardize,fieldList)
        return standardizedFields

    def _is_standardize_strict_mode(self,mergedFields, tableName):
        isStandardizeStrictMode = False
        standardizedFieldsStrict = self._get_standardized_field_strict(self.dictTables[tableName]['fieldName'],
                                                                       tableName)
        standardizedFieldsStrictPattern = '|'.join(standardizedFieldsStrict)
        if isinstance(standardizedFieldsStrictPattern, str) and isinstance(mergedFields, str):
            if standardizedFieldsStrictPattern != NULLSTR:
                matched = re.search(standardizedFieldsStrictPattern,mergedFields)
                if matched is not None:
                    isStandardizeStrictMode = True
        return isStandardizeStrictMode

    def _is_header_in_row(self,row,tableName):
        mergedRow = reduce(self._merge, row)
        headerStandardize = self.dictTables[tableName]['headerStandardize']
        isHeaderInRow = self._is_field_matched(headerStandardize, mergedRow)
        return isHeaderInRow

    def _is_field_first(self,tableName,firstField):
        #对获取到的字段做标准化(需要的话),然后和配置表中代表最后一个字段(或模式)做匹配,如匹配到,则认为找到表尾
        #对于现金分红情况表,因为字段为时间,则用模式去匹配,匹配到一个即可认为找到表尾
        fieldFirst = self.dictTables[tableName]["fieldFirst"]
        isFieldFirst = self._is_field_matched(fieldFirst,firstField)
        return isFieldFirst

    def _is_field_match_standardize(self, field, tableName):
        isFieldInList = False
        standardizedFields = self._get_standardized_field(self.dictTables[tableName]['fieldName'],tableName)
        aliasFields = list(self.dictTables[tableName]['fieldAlias'].keys())
        discardFields = list(self.dictTables[tableName]['fieldDiscard'])
        standardizedFields.extend(aliasFields)
        standardizedFields.extend(discardFields)
        standardizedFields = [field for field in standardizedFields if self._is_valid(field)]
        assert isinstance(standardizedFields,list),"patternList(%s) must be a list of string"%standardizedFields
        for pattern in standardizedFields:
            if self._is_field_matched(pattern,field):
                isFieldInList = True
                break
        return isFieldInList

    def _is_field_in_standardize_by_mode(self,field,isStandardizeStrict,tableName):
        if isStandardizeStrict == True:
            isFieldInStandardize = self._is_field_in_standardize_strict(field,tableName)
        else:
            isFieldInStandardize = self._is_field_in_standardize(field,tableName)
        return isFieldInStandardize

    def _is_field_in_standardize(self,field,tableName):
        # 把field按严格标准进行标准化,然后和判断该字段是否和同样方法标准化后的某个字段相同.
        isFieldInList = False
        standardizedFields = self._get_standardized_field(self.dictTables[tableName]['fieldName'],
                                                                       tableName)
        aliasFields = list(self.dictTables[tableName]['fieldAlias'].keys())
        discardFields = list(self.dictTables[tableName]['fieldDiscard'])
        standardizedFields.extend(aliasFields)
        standardizedFields.extend(discardFields)
        standardizedFields = [field for field in standardizedFields if self._is_valid(field)]
        assert isinstance(standardizedFields,
                          list), "patternList(%s) must be a list of string" % standardizedFields
        fieldStrict = self._get_standardized_field(field, tableName)

        if fieldStrict in standardizedFields:
            isFieldInList = True
        return isFieldInList

    def _is_field_in_standardize_strict(self, field,tableName):
        #把field按严格标准进行标准化,然后和判断该字段是否和同样方法标准化后的某个字段相同.
        isFieldInList = False
        standardizedFieldsStrict = self._get_standardized_field_strict(self.dictTables[tableName]['fieldName'],tableName)
        aliasFields = list(self.dictTables[tableName]['fieldAlias'].keys())
        discardFields = list(self.dictTables[tableName]['fieldDiscard'])
        standardizedFieldsStrict.extend(aliasFields)
        standardizedFieldsStrict.extend(discardFields)
        standardizedFieldsStrict = [field for field in standardizedFieldsStrict if self._is_valid(field)]
        assert isinstance(standardizedFieldsStrict,list),"patternList(%s) must be a list of string"%standardizedFieldsStrict
        fieldStrict = self._get_standardized_field_strict(field,tableName)

        if fieldStrict in standardizedFieldsStrict:
            isFieldInList = True
        return isFieldInList

    def _is_field_matched(self,pattern,field):
        isFieldMatched = False
        if isinstance(pattern, str) and isinstance(field, str):
            if pattern != NULLSTR:
                matched = re.search(pattern,field)
                if matched is not None:
                    isFieldMatched = True
        return isFieldMatched

    def _is_record_exist(self, conn, tableName, dataFrame):
        #用于数据在插入数据库之前,通过组合的关键字段判断记录是否存在.
        fieldFromHeader = self.dictTables[tableName]["fieldFromHeader"]
        primaryKey = [key for key,value in self.commonFileds.items() if value.find('NOT NULL') >= 0]
        #对于Sqlit3,字符串表示为'string' ,而不是"string".
        condition = ' and '.join([str(key) + '=\'' + str(dataFrame[key].values[0]) + '\'' for key in primaryKey])
        sql = 'select count(*) from {} where '.format(tableName) + condition
        if fieldFromHeader != NULLSTR:
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
        for tableName in self.tableNames:
            if tableName not in allTables:
                sql = " CREATE TABLE IF NOT EXISTS [%s] ( \n\t\t\t\t\t" % tableName
                for commonFiled, type in self.commonFileds.items():
                    sql = sql + "[%s] %s\n\t\t\t\t\t," % (commonFiled, type)
                #由表头转换生产的字段
                fieldFromHeader = self.dictTables[tableName]["fieldFromHeader"]
                if fieldFromHeader != "":
                    sql = sql + "[%s] VARCHAR(20)\n\t\t\t\t\t,"%fieldFromHeader
                sql = sql[:-1]  # 去掉最后一个逗号
                #创建新表
                standardizedFields = self._get_standardized_field(self.dictTables[tableName]['fieldName'],tableName)
                duplicatedFields = self._get_duplicated_field(standardizedFields)
                for fieldName in duplicatedFields:
                    if fieldName is not NaN:
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
    parser = DocParserSql(gConfig)
    parser.initialize()
    return parser