#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 12/9/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserSql.py
# @Note    : 用于sql数据库的读写

from interpreterAccounting.docparser.docParserBaseClass import  *
import sqlite3 as sqlite
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
from pandas import DataFrame
import sys
from importlib import reload
import time

#reload(sys)

#深度学习模型的基类
class DocParserSql(DocParserBase):
    def __init__(self,gConfig):
        super(DocParserSql, self).__init__(gConfig)
        #self.database = os.path.join(gConfig['working_directory'],gConfig['database'])
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

        #对数据进行预处理,两行并在一行的分开,去掉空格等
        dataframe = self._process_value_pretreat(dataframe,tableName)
        #针对合并所有者权益表的前三列空表头进行合并,对转置表进行预转置,使得其处理和其他表一致
        dataframe = self._process_header_merge_simple(dataframe, tableName)

        #把跨多个单元格的表字段名合并成一个
        dataframe = self._process_field_merge_simple(dataframe,tableName)

        #去掉空字段及无用字段
        dataframe = self._process_field_discard(dataframe, tableName)

        #去掉无用的表头;同时对水平表进行转置,把字段名由index转为column
        dataframe = self._process_header_discard(dataframe, tableName)

        #把表头进行标准化
        dataframe = self._process_header_standardize(dataframe,tableName)

        #把表字段名进行标准化
        dataframe = self._process_field_standardize(dataframe,tableName)

        #同一张表的相同字段在不同财务报表中名字不同,需要统一为相同名称,统一后再去重
        dataframe = self._process_field_alias(dataframe,tableName)

        #把表字段名统一命名后,再进行标准化之后去重复
        dataframe = self._process_field_standardize(dataframe,tableName)

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
            dataFrame = pd.DataFrame(table, index=None)
            countTotalFields = len(dataFrame.index.values)
        dataFrame.fillna(NONESTR,inplace=True)
        return dataFrame,countTotalFields

    def _write_to_sqlite3(self, dataFrame,tableName):
        conn = self._get_connect()
        for i in range(1,len(dataFrame.index)):
            sql_df = pd.DataFrame(dataFrame.iloc[i]).T
            sql_df.columns = dataFrame.iloc[0].values
            isRecordExist = self._is_record_exist(conn, tableName, sql_df)
            if not isRecordExist:
                sql_df.to_sql(name=tableName,con=conn,if_exists='append',index=False)
                conn.commit()
            else:
                self.logger.info("table %s is already exist in database!"%tableName)
        conn.close()

    def _rowPretreat(self,row):
        self.lastValue = None
        row = row.apply(self._valuePretreat)
        return row

    def _valuePretreat(self,value):
        try:
            if isinstance(value,str):
                if value != NONESTR and value != NULLSTR:
                    value = re.sub('不适用$',NULLSTR,value)
                    value = re.sub('元$',NULLSTR,value)#解决海螺水泥2018年报中,普通股现金分红情况表中出现中文字符,导致_process_field_merge出错
                    value = re.sub('^）\\s*',NULLSTR,value)
                    result = re.split("[ ]{2,}",value,maxsplit=1)
                    if len(result) > 1:
                        value,self.lastValue = result
                else:
                    if self.lastValue != None and value == NONESTR:
                        value = self.lastValue
                        self.lastValue = None
        except Exception as e:
            print(e)
        return value

    def _process_value_pretreat(self,dataFrame,tableName):
        #采用正则表达式替换空字符,对一个字段中包含两个数字字符串的进行拆分
        #解决奥美医疗2018年年报,主要会计数据中,存在两列数值并列到了一列,同时后接一个None的场景.
        #东材科技2018年年报,普通股现金分红流量表,表头有很多空格,影响_process_header_discard,需要去掉
        dataFrame.iloc[1:,1:]  = dataFrame.iloc[1:,1:].apply(self._rowPretreat,axis=1)
        dataFrame = dataFrame.apply(lambda row:row.apply(lambda x:x.replace(' ',NULLSTR)
                                                         .replace('(','（').replace(')','）')))
        return dataFrame

    def _process_header_merge_simple(self, dataFrame, tableName):
        isHorizontalTable = self.dictTables[tableName]['horizontalTable']
        mergedRow = None
        firstHeader = self.dictTables[tableName]['header'][0]
        lastIndex = 0
        # 增加blankFrame来驱动最后一个field的合并
        blankFrame = pd.DataFrame([''] * len(dataFrame.columns.values), index=dataFrame.columns).T
        dataFrame = dataFrame.append(blankFrame)
        for index, field in enumerate(dataFrame.iloc[:, 0]):
            isRowNotAnyNone = self._is_row_not_any_none(dataFrame.iloc[index])
            isHeaderInRow = self._is_header_in_row(dataFrame.iloc[index].tolist(),tableName)
            isHeaderInMergedRow = self._is_header_in_row(mergedRow,tableName)
            isRowAllInvalid = self._is_row_all_invalid(dataFrame.iloc[index])
            if isRowAllInvalid == False:
                if isHeaderInRow == False:
                    #表字段所在的行,清空合并行
                    if isHeaderInMergedRow:
                        if index > lastIndex + 1:
                            dataFrame.iloc[lastIndex] = mergedRow
                            dataFrame.iloc[lastIndex + 1:index] = NaN
                        mergedRow = None
                else:
                    if isHeaderInMergedRow == False:
                        #解决再升科技2018年年报,合并所有者权益变动表在每个分页中插入了表头
                        mergedRow = None
                if isRowNotAnyNone == True: #and isHeaderInRow == True:
                    #表头或表字段所在的起始行
                    if self._is_first_field_in_row(mergedRow, tableName):
                        if index > lastIndex + 1:
                            dataFrame.iloc[lastIndex] = mergedRow
                            dataFrame.iloc[lastIndex + 1:index] = NaN
                    if not (isHeaderInRow == True and isHeaderInMergedRow == True):
                        #解决大立科技：2018年年度报告,有一行", , , ,调整前,调整后, , , , ",满足isRowNotAnyNone==True条件,但是需要继续合并
                        mergedRow = None
            if mergedRow is None:
                mergedRow = dataFrame.iloc[index].tolist()
                lastIndex = index
            else:
                mergedRow = self._get_merged_row(dataFrame.iloc[index].tolist(), mergedRow, isFieldJoin=True)

        if isHorizontalTable == True:
            #如果是转置表,则在此处做一次转置,后续的处理就和非转置表保持一致了
            #去掉最后一行空行
            indexDiscardField = dataFrame.iloc[:, 0].isin(self._get_invalid_field())
            dataFrame.loc[indexDiscardField] = NaN
            dataFrame = dataFrame.dropna(axis=0)
            #把第一列做成索引
            dataFrame.set_index(0,inplace=True)
            dataFrame = dataFrame.T.copy()
        else:
            columns = dataFrame.iloc[0].copy()
            indexDiscardField = dataFrame.iloc[:, 0].isin([firstHeader])
            dataFrame.loc[indexDiscardField] = NaN
            dataFrame.columns = columns
            dataFrame = dataFrame.dropna(axis=0).copy()
        return dataFrame

    @loginfo()
    def _process_field_merge_simple(self,dataFrame,tableName):
        mergedRow = None
        lastIndex = 0
        mergedFields = reduce(self._merge,dataFrame.iloc[:,0].tolist())
        isStandardizeStrictMode = self._is_standardize_strict_mode(mergedFields,tableName)
        #增加blankFrame来驱动最后一个field的合并
        blankFrame = pd.DataFrame(['']*len(dataFrame.columns.values),index=dataFrame.columns).T
        dataFrame = dataFrame.append(blankFrame)

        for index,field in enumerate(dataFrame.iloc[:,0].tolist()):
            #识别新字段的起始行
            isRowNotAnyNone = self._is_row_not_any_none(dataFrame.iloc[index])
            isHeaderInRow = self._is_header_in_row(dataFrame.iloc[index].tolist(),tableName)
            if isRowNotAnyNone or isHeaderInRow:
                if self._is_field_match_standardize(field,tableName):
                    if index > lastIndex + 1 and mergedRow is not None:
                        # 把前期合并的行赋值到dataframe的上一行
                        dataFrame.iloc[lastIndex] = mergedRow
                        dataFrame.iloc[lastIndex + 1:index] = NaN
                    mergedRow = None
                else:
                    if isinstance(mergedRow,list):
                        mergedField = mergedRow[0]
                        if self._is_field_in_standardize_by_mode(mergedField, isStandardizeStrictMode, tableName):
                            if index > lastIndex + 1:
                                dataFrame.iloc[lastIndex] = mergedRow
                                dataFrame.iloc[lastIndex + 1:index] = NaN
                            mergedRow = None
                        elif isRowNotAnyNone == True and isHeaderInRow == False:
                            if mergedField == NULLSTR and self._is_header_in_row(mergedRow,tableName):
                                if index > lastIndex + 1:
                                    dataFrame.iloc[lastIndex] = mergedRow
                                    dataFrame.iloc[lastIndex + 1:index] = NaN
                                mergedRow = None
                        elif isRowNotAnyNone == True and isHeaderInRow == True:
                            mergedRow = None
            else:
                #if self._is_field_match_standardize(field,tableName):
                if self._is_field_in_standardize_by_mode(field, isStandardizeStrictMode, tableName):
                    if index > lastIndex + 1 and mergedRow is not None:
                        # 把前期合并的行赋值到dataframe的上一行
                        dataFrame.iloc[lastIndex] = mergedRow
                        dataFrame.iloc[lastIndex + 1:index] = NaN
                    if mergedRow is not None and mergedRow[0] != NULLSTR:
                        #解决大立科技2018年财报中主要会计数据解析不准确的问题,原因是总资产(元)前面接了一个空字段,空字段的行需要合并到下一行中
                        mergedRow = None

            if mergedRow is None:
                mergedRow = dataFrame.iloc[index].tolist()
                lastIndex = index
            else:
                mergedRow = self._get_merged_row(dataFrame.iloc[index].tolist(), mergedRow, isFieldJoin=True)
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
                    if isinstance(firstHeader,str) and (firstHeader == '分红年度' or firstHeader == '年度'):
                        #firstHeader == '年度'是为了解决海螺水泥2018年年报普通股现金分红情况表中,表头是年度
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
                    value = value.replace('\n', NULLSTR).replace(' ', NULLSTR).replace(NONESTR,NULLSTR)\
                        .replace('/',NULLSTR).replace('）',NULLSTR)
                    #解决迪安诊断2018年财报主要会计数据中,把最后一行拆为"归属于上市公司股东的净资产（元"和"）"
                    #高德红外2018年报,无效值用'--'填充,部分年报无效值用'-'填充
                    value = re.sub('.*-$',NULLSTR,value)
            except Exception as e:
                print(e)
            return value
        dataFrame.iloc[1:] = dataFrame.iloc[1:].apply(lambda x: x.apply(valueStandardize))
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

        standardizedFields = self._get_standardized_field(self.dictTables[tableName]['fieldName'], tableName)
        duplicatedFieldsStandard = self._get_duplicated_field(standardizedFields)

        dataFrame.iloc[0] = duplicatedFields
        duplicatedFieldsResult = []
        for field in duplicatedFields:
            if field in duplicatedFieldsStandard:
                duplicatedFieldsResult += [field]
            else:
                self.logger.warning('field %s is not exist in %s'%(field,tableName))
                #删除该字段
                indexDiscardField = dataFrame.iloc[0].isin([field])
                discardColumns = indexDiscardField[indexDiscardField == True].index.tolist()
                #dataFrame.T.loc[indexDiscardField] = NaN
                dataFrame[discardColumns] = NaN
                dataFrame = dataFrame.dropna(axis=1).copy()
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
        indexDiscardField = dataFrame.iloc[:,0].isin(fieldDiscard+self._get_invalid_field())
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
        #在标准化后,某些无用字段可能被标准化为NaN,需要去掉
        #dataFrame.loc[NaN] = NaN
        #dataFrame = dataFrame.dropna(axis=1).copy()
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

    def _is_first_field_in_row(self, row_or_field, tableName):
        #对获取到的字段做标准化(需要的话),然后和配置表中代表最后一个字段(或模式)做匹配,如匹配到,则认为找到表尾
        #对于现金分红情况表,因为字段为时间,则用模式去匹配,匹配到一个即可认为找到表尾
        if row_or_field == None:
            return False
        if isinstance(row_or_field, list):
            firstField = row_or_field[0]
        else:
            firstField = row_or_field
        fieldFirst = self.dictTables[tableName]["fieldFirst"]
        fieldFirst = '^' + fieldFirst
        isFirstFieldInRow = self._is_field_matched(fieldFirst, firstField)
        return isFirstFieldInRow

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
            #pattern = '^' + pattern
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
        standardizedFields = self._get_standardized_field(self.dictTables[tableName]['fieldName'],tableName)
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

    def _is_row_all_invalid(self,row:DataFrame):
        #如果该行以None开头,其他所有字段都是None或NULLSTR,则返回True
        isRowAllInvalid = False
        if (row == NULLSTR).all():
            #如果是空行,返回False,空行有特殊用途,一般加到最后一行来驱动前一个字段的合并
            return isRowAllInvalid
        mergedField = reduce(self._merge,row.tolist())
        #解决上峰水泥2017年中出现" ,None,None,None,None,None"的情况,以及其他年报中出现"None,,None,,None"的情况.
        isRowAllInvalid = not self._is_valid(mergedField)
        return isRowAllInvalid

    def _is_row_not_any_none(self,row:DataFrame):
        return (row != NONESTR).all()

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