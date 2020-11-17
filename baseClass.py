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
        self.debugIsOn = gConfig['debugIsOn'.lower()]
        self.program_directory = gConfig['program_directory']
        self.working_directory = os.path.join(self.gConfig['working_directory'], self._get_module_path())
        self.data_directory = gConfig['data_directory']
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
        #self.tablePrefixs =  list(set([self._get_table_prefix(reportType) for reportType in self.reportTypes]))


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


    def _set_dataset(self,index=None):
        if isinstance(index,list):
            self._data = [page for i, page in enumerate(self._data) if i in index]
            self._index = 0
            self._length = len(self._data)


    def _is_file_name_valid(self,fileName):
        assert fileName != None and fileName != NULLSTR, "filename (%s) must not be None or NULL" % fileName
        isFileNameValid = False
        #reportTypes = self.gJsonBase['报告类型']
        pattern = '|'.join(self.reportTypes)
        if isinstance(pattern, str) and isinstance(fileName, str):
            if pattern != NULLSTR:
                matched = re.search(pattern, fileName)
                if matched is not None:
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
        assert(typeLict,list),"parameter typeList must be a list!"
        type = defaultType
        for key in typeLict:
            match = re.search(local_name[key],value)
            if match is not None:
                type = key.split('_')[-1]
                break
        return type


    def _get_tablename_by_report_type(self, reportType, tableName):
        # 根据报告类型转换成相应的表名,如第一季度报告,合并资产负债表 转成 季报合并资产负债表
        assert reportType != NULLSTR, "reportType must not be NULL!"
        tablePrefix = self._get_tableprefix_by_report_type(reportType)
        return tablePrefix + tableName


    def _get_report_type_by_filename(self, name):
        #assert,因为repair_table会传进来一个文件 通用数据：适用所有年度报告.xlsx 不符合标准文件名
        #assert self._is_matched('\\d+年',name),"name(%s) is invalid"%name
        type = name
        pattern = "\\d+年([\\u4E00-\\u9FA5]+)"
        matched = re.findall(pattern,name)
        if matched is not None and len(matched) > 0:
            type = matched.pop()
        #reportType = self._alias(type, self.reportTypeAlias)
        reportType = self._get_report_type_alias(type)
        return reportType


    def _get_path_by_report_type(self, type):
        #reportTypes = self.gJsonBase['报告类型']
        assert type in self.reportTypes, "type(%s) is invalid ,which not in [%s] "%(type,self.reportTypes)
        path = os.path.join(self.data_directory,type)
        if not os.path.exists(path):
            os.mkdir(path)
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


    def _get_connect(self):
        #用于获取数据库连接
        return sqlite.connect(self.database)

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
