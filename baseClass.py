#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/25/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserBaseClass.py
from six import unichr

from loggerClass import *
import functools
import numpy as np
import os
import re
import sqlite3 as sqlite
import pysnooper
import time
from datetime import date,timedelta
from pandas import DataFrame
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
#数据读写处理的基类

NULLSTR = ''
NONESTR = 'None'
NaN = np.nan
EOF = 'EOF）'  #加）可解决fidller


class BaseClass():
    def __init__(self,gConfig):
        self.gConfig = gConfig
        self.gJsonInterpreter = gConfig['gJsonInterpreter'.lower()]
        self.gJsonBase = gConfig['gJsonBase'.lower()]
        #self.debugIsOn = gConfig['debugIsOn'.lower()]
        self.program_directory = gConfig['program_directory']
        self.working_directory = os.path.join(self.gConfig['working_directory'], self._get_module_path())
        self.logging_directory = os.path.join(self.gConfig['logging_directory'], self._get_module_path())
        self.data_directory = gConfig['data_directory']
        self.stockcodefile = os.path.join(self.data_directory,self.gConfig['stockcodefile'])
        self.unitestIsOn = gConfig['unittestIsOn'.lower()]
        self._data = list()
        self._index = 0
        self._length = len(self._data)
        #不同的类继承BaseClass时,logger采用不同的名字
        self._logger = Logger(gConfig,self._get_class_name(gConfig)).logger
        self.database = os.path.join(gConfig['working_directory'],gConfig['database'])
        self.reportTypeAlias = self.gJsonBase['reportTypeAlias']
        self.reportTypes =  self.gJsonBase['reportType']
        self.companyAlias = self.gJsonBase['companyAlias']
        # 编译器,文件解析器共同使用的关键字
        self.commonFields = self.gJsonBase['公共表字段定义']
        self.tableNames = self.gJsonBase['TABLE'].split('|')
        self.dictTables = {keyword: value for keyword,value in self.gJsonBase.items() if keyword in self.tableNames}
        self.filenameAlias = self.gJsonBase['filenameAlias']
        #self._get_interpreter_keyword()
        #self._create_tables(self.tableNames)


    def __iter__(self):
        return self


    def __next__(self):
        try:
            data = self._data[self._index]
        except IndexError:
            raise StopIteration
        self._index += 1
        return data


    def __getitem__(self, item):
        return self._data[item]


    def _get_connect(self):
        #用于获取数据库连接
        return sqlite.connect(self.database)


    def _get_interpreter_keyword(self):
        # 编译器,文件解析器共同使用的关键字
        #self.tableNames = self.gJsonBase['TABLE'].split('|')
        #self.commonFields = self.gJsonBase['公共表字段定义']
        #self.dictTables = {keyword: value for keyword,value in self.gJsonBase.items() if keyword in self.tableNames}
        ...


    def _get_filename_alias(self,filename):
        aliasedFilename = self._alias(filename, self.filenameAlias)
        return aliasedFilename


    def _get_dict_tables(self,tableNames,dictTablesBase):
        """
            该函数目前只用在interpreterNature,interpreterCrawl,interpreterAnalysize中, 这些解释器用到了interpreterBase.json中配置的公共表字段定义
            而interpreterAccounting不需要调用该函数, 他用了一套独立的 公共表字段定义
            args:
                tableNames - 当前解释器下所能读取到的表名列表, 一般配置在 interpreterXXXX.json的 TABLE 关键字下
                dictTablesBase - 定义在interpreterBase.json下的 通用表配置
            reutrn:
                dictTables - 当前interpreterXXXX.json配置的表和interpreterBase.json配置的表进行融合, 融合的规则:
                '''
                1) dictTablesBase.keys()中的表名和tableNames中的表名重合,则从dictTablesBase中取出该表配置放到dictTable中;
                2) 当前解释器配置文件interpreterXXXX.json中配置的表配置, 更新到第一步的dictTable中
                '''
        """
        dictTables = {keyword: value for keyword, value in dictTablesBase.items() if
                           keyword in tableNames}
        dictTables.update({keyword: value for keyword, value in self.gJsonInterpreter.items() if
                           keyword in tableNames})
        # 如果表的配置中,还有 parent这段,则要和父表的字段进行合并,合并的原则: 1) 子表的value是一个值,覆盖附表；2)value是列表,则追加到父表；3)value是dict,则进行递归
        for tableName, dictTable in dictTables.items():
            parent = dictTable['parent']
            if parent != NULLSTR:
                mergedDictTable = self._merged_dict_table(dictTable,dictTables[parent])
                dictTables[tableName].update(mergedDictTable)
        return dictTables


    def _merged_dict_table(self,dictTable,dictTableParent):
        """
        args:
            dictTable - 当前表的配置参数
            dictTableParent - 父表的配置参数
        reutrn:
            dictTableMerged - 当前表和父表融合后的配置, 融合的规则:
                '''
                1) 当前表的value是一个值, 则覆盖父表;
                2) 当前表的value是一个list,则追加到父表；
                3) 当前表的value是一个dict,则进行递归调用;
                '''
        """
        dictTableMerged = dictTableParent.copy()
        for key,value in dictTable.items():
            # 遍历子表的值, 和父表进行合并
            if isinstance(value, list):
                # 如果子表的值是列表,则追加到父表
                if len(value) != 0:
                    dictTableMerged.setdefault(key,[]).append(*value)
            elif isinstance(value, dict):
                # 如果子表的值是dict, 则进行递归
                dictTableMergedChild = self._merged_dict_table(value, dictTableMerged[key])
                dictTableMerged.update({key: dictTableMergedChild})
            else:
                # 如果子表的值非上述几中,则覆盖父表
                if key != 'parent':
                    # 避免迭代循环
                    dictTableMerged.update({key: value})
        return dictTableMerged


    def _set_dataset(self,index=None):
        if isinstance(index,list):
            self._data = [page for i, page in enumerate(self._data) if i in index]
            self._index = 0
            self._length = len(self._data)


    def _is_file_name_valid(self,fileName):
        assert fileName != None and fileName != NULLSTR, "filename (%s) must not be None or NULL" % fileName
        isFileNameValid = False
        #reportTypes = self.gJsonBase['报告类型']
        #pattern = '|'.join(self.reportTypes)
        type = self._get_report_type_by_filename(fileName)
        #if isinstance(pattern, str) and isinstance(fileName, str):
        #    if pattern != NULLSTR:
        #        matched = re.search(pattern, fileName)
        #        if matched is not None:
        #            isFileNameValid = True
        if type != NULLSTR:
            isFileNameValid = True
        return isFileNameValid


    def _get_tableprefix_by_report_type(self, reportType):
        assert reportType != NULLSTR,"reportType must not be NULL!"
        tablePrefix = NULLSTR
        dictTablePrefix = self.gJsonBase['tablePrefix']
        if reportType in dictTablePrefix.keys():
            tablePrefix = dictTablePrefix[reportType]
        else:
            self.logger.error('reportType(%s) is invalid,it must one of %s'%(reportType,dictTablePrefix.keys()))
        return tablePrefix


    def _get_token_type(self, local_name,value,typeLict,defaultType):
        #根据传入的TypeList,让lexer从defaultType中进一步细分出所需的type(从TypeList中选出)
        #Local_name中保存了每个Type所对应的正则表达式
        #VALUE为lexer所识别的值
        assert isinstance(typeLict,list),"parameter typeList must be a list!"
        type = defaultType
        for key in typeLict:
            match = re.search(local_name[key],value)
            if match is not None:
                type = key.split('_')[-1]
                break
        return type


    def _get_company_time_type_code_by_filename(self, filename):
        #timeStandardize = self.gJsonBase['timeStandardize'] # '\\d+年'
        #time = self._standardize('\\d+年',filename)
        #time = self._standardize(timeStandardize, filename)
        time = self._get_time_by_filename(filename)
        type = self._get_report_type_by_filename(filename)
        company,code = self._get_company_code_by_content(filename)
        return company,time,type,code


    def _get_time_by_filename(self,filename):
        timeStandardize = self.gJsonBase['timeStandardize'] # '\\d+年'
        #time = self._standardize('\\d+年',filename)
        time = self._standardize(timeStandardize, filename)
        return time


    def _get_company_code_by_content(self,content):
        codeStandardize = self.gJsonBase['codeStandardize'] # （\\d+）
        #code = self._standardize('（\\d+）',content)
        code = self._standardize(codeStandardize, content)
        if code is not NaN:
            code = code.replace('（',NULLSTR).replace('）',NULLSTR)
        companyStandardize = self.gJsonBase['companyStandardize']   # "[*A-Z]*[\\u4E00-\\u9FA5]+[A-Z0-9]*"
        #company = self._standardize("[*A-Z]*[\\u4E00-\\u9FA5]+[A-Z0-9]*",content)
        company = self._standardize(companyStandardize, content)
        return company,code


    def _get_tablename_by_report_type(self, reportType, tableName):
        # 根据报告类型转换成相应的表名,如第一季度报告,合并资产负债表 转成 季报合并资产负债表
        assert reportType != NULLSTR, "reportType must not be NULL!"
        tablePrefix = self._get_tableprefix_by_report_type(reportType)
        return tablePrefix + tableName


    def _get_report_type_by_filename(self, filename):
        #assert,因为repair_table会传进来一个文件 通用数据：适用所有年度报告.xlsx 不符合标准文件名
        #assert self._is_matched('\\d+年',name),"name(%s) is invalid"%name
        #type = filename
        reportType = NULLSTR
        #pattern = "\\d+年([\\u4E00-\\u9FA5]+)"
        reportTypeStandardize = self.gJsonBase['reportTypeStandardize']  # "\\d+年([\\u4E00-\\u9FA5]+)"
        matched = re.findall(reportTypeStandardize, filename)
        if matched is not None and len(matched) > 0:
            type = matched.pop()
            #reportType = self._alias(type, self.reportTypeAlias)
            reportType = self._get_report_type_alias(type)
        return reportType


    def _get_path_by_report_type(self, type):
        #reportTypes = self.gJsonBase['报告类型']
        #assert type in self.reportTypes, "type(%s) is invalid ,which not in [%s] "%(type,self.reportTypes)
        path = NULLSTR
        if type in self.reportTypes:
            path = os.path.join(self.data_directory,type)
            if not os.path.exists(path):
                os.mkdir(path)
        else:
            self.logger.error("type(%s) is invalid ,which not in [%s] " % (type, self.reportTypes))
        return path


    def _get_path_by_filename(self, filename):
        type = self._get_report_type_by_filename(filename)
        #type = self._get_report_type_alias(type)
        path = self._get_path_by_report_type(type)
        return path


    def _get_text(self,page):
        return page


    def _standardize(self,fieldStandardize,field):
        standardizedField = field
        if isinstance(field, str) and isinstance(fieldStandardize, str) and fieldStandardize !="":
            matched = re.search(fieldStandardize, field)
            if matched is not None:
                standardizedField = matched[0]
            else:
                standardizedField = NaN
        return standardizedField


    def _strQ2B(self,ustring):
        """把字符串全角转半角"""
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
                rstring += uchar
            else:
                rstring += unichr(inside_code)
        return rstring


    def _strB2Q(self,ustring):
        """把字符串半角转全角"""
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x0020:  # 除了空格其他的全角半角的公式为:半角=全角-0xfee0
                inside_code = 0x3000
            else:
                inside_code += 0xfee0
            if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角字符就返回原来的字符
                rstring += uchar
            else:
                rstring += unichr(inside_code)
        return rstring


    def _merge_table(self, dictTable=None,interpretPrefix=None):
        if dictTable is None:
            dictTable = list()
        return dictTable


    def _get_report_type_alias(self, reportType):
        aliasedReportType = NULLSTR
        reportTypeTotal = set(list(self.reportTypeAlias.keys()) + self.reportTypes)
        if reportType in reportTypeTotal:
            aliasedReportType = self._alias(reportType, self.reportTypeAlias)
        return aliasedReportType


    def _get_company_alias(self,company):
        aliasedCompany = self._alias(company,self.companyAlias)
        return aliasedCompany


    def _alias(self, name, dictAlias):
        alias = name
        aliasKeys = dictAlias.keys()
        if len(aliasKeys) > 0:
            if name in aliasKeys:
                alias = dictAlias[name]
        return alias


    def _write_table(self,tableName,table):
        pass


    def _close(self):
        pass


    def _debug_info(self):
        pass


    def _get_class_name(self,*args):
        return 'Base'


    def _get_module_path(self):
        module = self.__class__.__module__
        path = os.path.join(*module.split('.'))
        return path


    def _write_to_sqlite3(self,dataFrame:DataFrame, tableName):
        conn = self._get_connect()
        dataFrame.to_sql(tableName, conn, if_exists='replace', index=False)
        conn.close()


    def _get_time_now(self):
        return time.strftime('%Y%m%d')


    def _get_last_week_day(self):
        now = date.today()
        if now.isoweekday() == 7:
            dayStep = 2
        elif now.isoweekday() == 6:
            dayStep = 2
        else:
            dayStep = 0
        #print(dayStep)
        lastWorkDay = now - timedelta(days=dayStep)
        lastWorkDay = lastWorkDay.strftime('%Y%m%d')
        return lastWorkDay


    @pysnooper.snoop()
    def _sql_executer(self,sql):
        conn = self._get_connect()
        try:
            conn.execute(sql)
            conn.commit()
            self.logger.debug('success to execute sql(脚本执行成功):\n%s' % sql)
        except Exception as e:
            # 回滚
            conn.rollback()
            self.logger.error('failed to execute sql(脚本执行失败):%s\n%s' % (str(e),sql))
        conn.close()


    def _is_matched(self,pattern,field):
        isMatched = False
        if isinstance(field, str) and isinstance(pattern, str) and pattern != NULLSTR:
            matched = re.search(pattern, field)
            if matched is not None:
                isMatched = True
        return isMatched


    def _get_merged_columns(self,tableName):
        mergedColumns = [key for key in self.commonFields.keys() if key != "ID"]
        mergedColumns = mergedColumns + self.dictTables[tableName]['fieldName']
        return mergedColumns


    def _create_tables(self,tableNames):
        # 用于想sqlite3数据库中创建新表
        conn = self._get_connect()
        cursor = conn.cursor()
        allTables = self._fetch_all_tables(cursor)
        allTables = list(map(lambda x: x[0], allTables))
        for tableName in tableNames:
            targetTableName =  tableName
            if targetTableName not in allTables:
                sql = " CREATE TABLE IF NOT EXISTS [%s] ( \n\t\t\t\t\t" % targetTableName
                for commonFiled, type in self.commonFields.items():
                    sql = sql + "[%s] %s\n\t\t\t\t\t," % (commonFiled, type)
                # 由表头转换生产的字段
                fieldFromHeader = self.dictTables[tableName]["fieldFromHeader"]
                if len(fieldFromHeader) != 0:
                    for field in fieldFromHeader:
                        sql = sql + "[%s] VARCHAR(20)\n\t\t\t\t\t," % field
                sql = sql[:-1]  # 去掉最后一个逗号
                # 创建新表
                standardizedFields = self.dictTables[tableName]['fieldName']
                #duplicatedFields = self._get_duplicated_field(standardizedFields)
                duplicatedFields = standardizedFields
                for fieldName in duplicatedFields:
                    if fieldName is not NaN:
                        if 'fieldType' in self.dictTables[tableName].keys() \
                            and fieldName in self.dictTables[tableName]['fieldType'].keys():
                            type = self.dictTables[tableName]['fieldType'][fieldName]
                        else:
                            type = 'NUMERIC'
                        sql = sql + "\n\t\t\t\t\t,[%s]  %s" % (fieldName, type)
                sql = sql + '\n\t\t\t\t\t)'
                try:
                    conn.execute(sql)
                    conn.commit()
                    print('创建数据库表%s成功' % (targetTableName))
                except Exception as e:
                    # 回滚
                    conn.rollback()
                    print(e, ' 创建数据库表%s失败' % targetTableName)

                # 创建索引
                sql = "CREATE INDEX IF NOT EXISTS [%s索引] on [%s] (\n\t\t\t\t\t" % (targetTableName, targetTableName)
                sql = sql + ", ".join(str(field) for field, value in self.commonFields.items()
                                      if value.find('NOT NULL') >= 0)
                sql = sql + '\n\t\t\t\t\t)'
                try:
                    conn.execute(sql)
                    conn.commit()
                    print('创建数据库%s索引成功' % (targetTableName))
                except Exception as e:
                    # 回滚
                    conn.rollback()
                    print(e, ' 创建数据库%s索引失败' % targetTableName)
        cursor.close()
        conn.close()


    def _fetch_all_tables(self, cursor):
        #获取数据库中所有的表,用于判断待新建的表是否已经存在
        try:
            cursor.execute("select name from sqlite_master where type='table' order by name")
        except Exception as e:
            print(e)
        return cursor.fetchall()


    def _is_record_exist(self, conn, tableName, dataFrame:DataFrame,specialKeys = None):
        #用于数据在插入数据库之前,通过组合的关键字段判断记录是否存在.
        #对于Sqlit3,字符串表示为'string' ,而不是"string".
        isRecordExist = False
        condition = self._get_condition(dataFrame,specialKeys)
        if condition == NULLSTR:
            #condition为空时,说明dataFrame没有有效数据,直接返回False
            return isRecordExist
        sql = 'select count(*) from {} where '.format(tableName) + condition
        result = conn.execute(sql).fetchall()
        if len(result) > 0:
            isRecordExist = (result[0][0] > 0)
        return isRecordExist


    def _get_condition(self,dataFrame,specialKeys = None):
        primaryKey = [key for key, value in self.commonFields.items() if value.find('NOT NULL') >= 0]
        if specialKeys is not None and isinstance(specialKeys,list):
            primaryKey = primaryKey + specialKeys
        # 对于Sqlit3,字符串表示为'string' ,而不是"string".
        joined = list()
        for key in primaryKey:
            if dataFrame[key].shape[0] == 0:
                joined = list()
                break
            current = '(' + ' or '.join(['{} = \'{}\''.format(key,value) for value in set(dataFrame[key].tolist())]) + ')'
            joined = joined + list([current])
        condition = NULLSTR
        if len(joined) > 0:
            condition = ' and '.join(joined)
        return condition


    def _sql_executer_script(self,sql):
        isSuccess = False
        conn = self._get_connect()
        #cursor = conn.cursor()
        try:
            conn.executescript(sql)
            conn.commit()
            self.logger.info('success to execute sql(脚本执行成功)!')
            isSuccess = True
        except Exception as e:
            # 回滚
            conn.rollback()
            self.logger.error('failed to execute sql(脚本执行失败):%s' % (str(e)))
        #cursor.close()
        conn.close()
        return isSuccess


    def _get_file_context(self,fileName):
        file_object = open(fileName,encoding='utf-8')
        file_context = NULLSTR
        try:
            file_context = file_object.read()  # file_context是一个string，读取完后，就失去了对test.txt的文件引用
        except Exception as e:
            self.logger.error('读取文件(%s)失败:%s' % (fileName,str(e)))
        finally:
            file_object.close()
        return file_context


    def _get_stock_list(self, companyList):
        assert isinstance(companyList,list),"Parameter companyList (%s) must be a list" % type(companyList)
        stockList = []
        stockcodeSpecial = [[company,code] for  company,code in self.gJsonBase['stockcode'].items()]
        if len(companyList) == 0:
            return stockList
        stockcodeHeader = self.gJsonBase['stockcodeHeader']
        if os.path.exists(self.stockcodefile):
            dataFrame = pd.read_csv(self.stockcodefile,names=stockcodeHeader,dtype=str)
            # stockcodeSpecial只有两列, 而stockcodeHeader有三列,最后一列为公司, 需要特殊处理
            dataFrameSpecial = pd.DataFrame(stockcodeSpecial,columns=stockcodeHeader[0:-1])
            dataFrameSpecial[stockcodeHeader[-1]] = '公司'
            dataFrame = dataFrame.append(dataFrameSpecial)
        else:
            dataFrame = pd.DataFrame(stockcodeSpecial,columns=stockcodeHeader)
        dataFrame = dataFrame.drop_duplicates()
        indexNeeded = dataFrame[stockcodeHeader[0]].isin(companyList)
        dataFrame = dataFrame[indexNeeded]
        stockList = dataFrame.values.tolist()
        companyDiffer = set(companyList).difference(set([company for company,_,_ in stockList]))
        if len(companyDiffer) > 0:
            self.logger.info("failed to get these stock list:%s"%companyDiffer)
        return stockList


    def _year_plus(self,reportTime, plusNumber):
        # 2019年 + 1 = 2020年, 2020年 - 1 = 2019年
        assert isinstance(reportTime,str) and isinstance(plusNumber,int),'reportTime must be str and plusNumber must be int!'
        newYear = int(reportTime.split('年')[0]) + plusNumber
        newYear = str(newYear) + '年'
        return newYear


    @property
    def index(self):
        return self._index - 1


    @property
    def logger(self):
        return self._logger


    def loginfo(text = NULLSTR):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self,*args, **kwargs):
                result = func(self,*args, **kwargs)
                self._logger.info('%s %s() %s:\n\t%s' % (text, func.__name__, list([*args]), result))
                return result
            return wrapper
        return decorator
