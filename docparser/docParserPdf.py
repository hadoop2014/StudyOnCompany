#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 12/9/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserPdf.py
# @Note    : 用于年报,半年报,季报pdf文件的读写

from docparser.docParserBaseClass import *
import pdfplumber
import pandas as pd
from openpyxl import Workbook

class DocParserPdf(DocParserBase):
    def __init__(self,gConfig):
        super(DocParserPdf, self).__init__(gConfig)
        self._interpretPrefix = NULLSTR
        self.table_settings = gConfig["table_settings"]
        self._load_data()

    def _load_data(self,input=None):
        self._pdf = pdfplumber.open(self.sourceFile,password='')
        self._data = self._pdf.pages
        self._index = 0
        self._length = len(self._data)

    def _get_text(self,page=None):
        #interpretPrefix用于处理比如合并资产负债表分布在多个page页面的情况
        #用于模拟文件结束符EOF,在interpretAccounting中单一个fetchtable语句刚好在文件尾的时候,解释器会碰到EOF缺失错误,所以在每一个page后补充EOF规避问题.
        if self._index == 1 :
            #解决贵州茅台年报中,贵州茅台酒股份有限公司2018 年年度报告,被解析成"贵州茅台酒股份有限公司 年年度报告 2018
            pageText = page.extract_text(y_tolerance=4)
        else:
            pageText = page.extract_text()
        if pageText is not None:
            pageText = self._interpretPrefix + pageText + self.EOF
        else:
            #千禾味业：2019年度审计报告.PDF文件中全部是图片,没有文字,需要做特殊处理
            pageText = self._interpretPrefix + self.EOF
            self.logger.error('the %s page %d\'s text of is be None' % (self.sourceFile,self._index))
        return pageText

    def _get_tables(self,dictTable):
        page = self.__getitem__(self._index-1)
        '''
        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "explicit_vertical_lines": [],
            "explicit_horizontal_lines": [],
            #"snap_tolerance": DEFAULT_SNAP_TOLERANCE,
            #"join_tolerance": DEFAULT_JOIN_TOLERANCE,
            #"edge_min_length": 3,
            #"min_words_vertical": DEFAULT_MIN_WORDS_VERTICAL,
            #"min_words_horizontal": DEFAULT_MIN_WORDS_HORIZONTAL,
            #"keep_blank_chars": False,
            #"text_tolerance": 3,
            #"text_x_tolerance": 3,
            #"text_y_tolerance": 3,
            #"intersection_tolerance": 3,
            #"intersection_x_tolerance": 3,
            #"intersection_y_tolerance": 3,

        }
        '''
        table_settings = self._get_table_settings(dictTable)
        return page.extract_tables(table_settings=table_settings)
        #return page.extract_tables()

    def _get_table_settings(self,dictTable):
        def valueTransfer(key,value):
            if key not in ["vertical_strategy","horizontal_strategy","explicit_vertical_lines","explicit_horizontal_lines",
                           "keep_blank_chars","intersection_x_tolerance","intersection_y_tolerance"]:
                value = int(value)
            elif key in ["keep_blank_chars"]:
                value = (value.lower() == "true")
            elif key in ["intersection_x_tolerance","intersection_y_tolerance"]:
                if value == "None":
                    value = None
                else:
                    value = int(value)
            else:
                value = str(value)
            return value
        table_settings = dict([(key,valueTransfer(key,value)) for key,value in self.table_settings.items()])
        table_settings = self._get_special_settings(dictTable,table_settings)
        return table_settings

    def _get_special_settings(self,dictTable,table_settings):
        keyName = '默认值'
        if dictTable['股票简称'] != NULLSTR:
            keyName = dictTable['股票简称']
            if keyName in self.gJsonBase['table_settings'].keys():
                if dictTable['报告时间'] == self.gJsonBase['table_settings'][keyName]['报告时间'] \
                    and dictTable['报告类型'] == self.gJsonBase['table_setttings'][keyName]['报告类型']:
                    table_settings.update({"snap_tolerance":
                                           self.gJsonBase['table_settings'][keyName]["snap_tolerance"]})
        else:
            table_settings.update({"snap_tolerance":
                                       self.gJsonBase['table_settings'][keyName]["snap_tolerance"]})
        return table_settings

    def _merge_table(self, dictTable=None,interpretPrefix=NULLSTR):
        assert dictTable is not None,"dictTable must not be None"
        self.interpretPrefix = interpretPrefix
        if dictTable['tableBegin'] == False:
            return dictTable
        savedTable = dictTable['table']
        tableName = dictTable['tableName']
        fetchTables = self._get_tables(dictTable)
        page_numbers = dictTable['page_numbers']
        processedTable,isTableEnd = self._process_table(fetchTables, tableName)
        dictTable.update({'tableEnd':isTableEnd})
        if isinstance(savedTable, list):
            savedTable.extend(processedTable)
        else:
            savedTable = processedTable

        if dictTable['tableBegin'] == True and dictTable['tableEnd'] == True:
            self.interpretPrefix = NULLSTR

        dictTable.update({'table':savedTable})
        return dictTable

    def _process_table(self,tables,tableName):
        processedTable,isTableEnd = NULLSTR , False
        if len(tables) == 0:
            return processedTable,isTableEnd
        #processedTable = [list(map(lambda x:str(x).replace('\n',NULLSTR).replace(' ',NULLSTR),row)) for row in tables[-1]]
        processedTable = [list(map(lambda x: str(x).replace('\n', NULLSTR), row)) for row in tables[-1]]
        fieldList = [row[0] for row in processedTable]
        mergedFields = reduce(self._merge,fieldList)
        if mergedFields == NULLSTR:
            processedTable = [row[1:] for row in processedTable]
            fieldList = [row[0] for row in processedTable]
            mergedFields = reduce(self._merge,fieldList)
        isTableEnd = self._is_table_end(tableName,mergedFields)
        if isTableEnd == True or len(tables) == 1:
            return processedTable, isTableEnd

        processedTable = NULLSTR
        for index,table in enumerate(tables):
            #table = [list(map(lambda x: str(x).replace('\n', NULLSTR).replace(' ',NULLSTR), row)) for row in table]
            table = [list(map(lambda x: str(x).replace('\n', NULLSTR), row)) for row in table]
            fieldList = [row[0] for row in table]
            headerList = table[0]
            mergedFields = reduce(self._merge, fieldList)
            mergedHeaders = reduce(self._merge,headerList)
            #if mergedFields == NULLSTR:
            #浙江鼎力2018年年报,分季度主要财务数据,表头单独在一页中,而表头的第一个字段刚好为空,因此不能做这个空字符串的判断.
            #    table = [row[1:] for row in table]
            #    fieldList = [row[0] for row in table]
            #    mergedFields = reduce(self._merge,fieldList)
            isTableEnd = self._is_table_end(tableName, mergedFields)
            isTableStart = self._is_table_start(tableName,mergedFields,mergedHeaders)
            if isTableStart == True:
                processedTable = table

            if isTableEnd == True:
                processedTable = table
                break
        return processedTable,isTableEnd

    def _is_table_start(self,tableName,mergedFields,mergedHeaders):
        #针对合并所有者权益表,第一个表头"项目",并不是出现在talbe[0][0],而是出现在第一列的第一个有效名称中
        isTableStart = False
        #firstHeaderName = self.dictTables[tableName]['header'][0]
        #headerList = [row[0] for row in table]
        #for header in headerList:
        #    if self._is_valid(header):
        #        if firstHeaderName == header:
        #            isTableStart = True
        #        break
        headerFirst = self.dictTables[tableName]["header"][0]
        fieldFirst = self.dictTables[tableName]['fieldFirst']
        assert fieldFirst != NULLSTR,'the first field of %s must not be NULL'%tableName
        if headerFirst == NULLSTR:
            #fieldFirst = fieldFirst.replace('(', '（').replace(')', '）')
            #headerFirst = '^' + fieldFirst
            headerFirst = self._get_standardized_header(self.dictTables[tableName]['header'][1],tableName)
            headerFirst = headerFirst.replace('(', '（').replace(')', '）')
            assert headerFirst != NULLSTR,'the second header of %s must not be NULL'%tableName
            if isinstance(mergedHeaders, str) and isinstance(headerFirst, str) and headerFirst != NULLSTR:
                mergedHeaders = mergedHeaders.replace('(', '（').replace(')', '）')
                matched = re.search('^' + headerFirst, mergedHeaders)
                if matched is not None:
                    isTableStart = True
        #else:
        headerFirst = headerFirst.replace('(', '（').replace(')', '）')  # 在正则表达式中,'()'是元符号,需要替换成中文符号
        fieldFirst = fieldFirst.replace('(', '（').replace(')', '）')
        fieldFirst = '^' + fieldFirst
        headerFirst = '^' + headerFirst
        headerFirst = '|'.join([headerFirst,fieldFirst])
        if isinstance(mergedFields, str) and isinstance(headerFirst, str) and headerFirst != NULLSTR:
            mergedFields = mergedFields.replace('(', '（').replace(')', '）')
            matched = re.search(headerFirst, mergedFields)
            if matched is not None:
                isTableStart = True
        return isTableStart

    def _is_table_end(self,tableName,mergedField):
        #对获取到的字段做标准化(需要的话),然后和配置表中代表最后一个字段(或模式)做匹配,如匹配到,则认为找到表尾
        #对于现金分红情况表,因为字段为时间,则用模式去匹配,匹配到一个即可认为找到表尾
        #针对合并所有者权益表,表尾字段"四、本期期末余额",并不是出现在talbe[-1][0],而是出现在第一列的最后两个字段,且有可能是分裂的
        isTableEnd = False
        fieldLast = self.dictTables[tableName]["fieldLast"]
        fieldLast = fieldLast.replace('(','（').replace(')','）')  #在正则表达式中,'()'是元符号,需要替换成中文符号
        fieldLast = '|'.join([field + '$' for field in fieldLast.split('|')])
        if isinstance(mergedField,str) and isinstance(fieldLast,str) and fieldLast != NULLSTR:
            mergedField = mergedField.replace('(','（').replace(')','）')
            matched = re.search(fieldLast,mergedField)
            if matched is not None:
                isTableEnd = True
        return isTableEnd

    def _close(self):
        self._pdf.close()

    @property
    def interpretPrefix(self):
        return self._interpretPrefix

    @interpretPrefix.setter
    def interpretPrefix(self,prefix):
        assert isinstance(prefix,str),"para(%s) of set_interpretPrefix must be string"%value
        self._interpretPrefix = prefix

    def initialize(self):
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        self.clear_logging_directory(self.logging_directory)
        suffix = self.sourceFile.split('.')[-1]
        assert suffix.lower() in self.gConfig['pdfSuffix'.lower()], \
            'suffix of {} is invalid,it must one of {}'.format(self.sourceFile, self.gConfig['pdfSuffix'.lower()])

def create_object(gConfig):
    parser=DocParserPdf(gConfig)
    parser.initialize()
    return parser
