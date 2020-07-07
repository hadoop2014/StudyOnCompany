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
        self.interpretPrefix = ''
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
        pageText = self.interpretPrefix + page.extract_text() + self.EOF
        return pageText

    def _get_tables(self,page = None):
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
            #elif key in ["explicit_vertical_lines","explicit_horizontal_lines"]:
            #    value = list([value.split('[')[-1].split(']')[0].split(',')])
            else:
                value = str(value)
            return value
        table_settings = dict([(key,valueTransfer(key,value)) for key,value in self.table_settings.items()])
        #return page.extract_tables(table_settings=table_settings)
        return page.extract_tables()

    def _merge_table(self, dictTable=None,interpretPrefix=''):
        assert dictTable is not None,"dictTable must not be None"
        self.interpretPrefix = interpretPrefix
        if dictTable['tableBegin'] == False:
            return dictTable
        savedTable = dictTable['table']
        tableName = dictTable['tableName']
        fetchTables = self._get_tables()
        page_numbers = dictTable['page_numbers']
        processedTable,isTableEnd = self._process_table(fetchTables, tableName, page_numbers)
        dictTable.update({'tableEnd':isTableEnd})
        if isinstance(savedTable, list):
            savedTable.extend(processedTable)
        else:
            savedTable = processedTable

        if dictTable['tableBegin'] == True and dictTable['tableEnd'] == True:
            self.interpretPrefix = ''

        dictTable.update({'table':savedTable})
        return dictTable

    def _process_table(self,tables,tableName,page_numbers):
        processedTable = [list(map(lambda x:str(x).replace('\n',''),row)) for row in tables[-1]]
        fieldList = [row[0] for row in processedTable]
        mergedFields = reduce(self._merge,fieldList)
        #isTableEnd = self._is_table_end(tableName,processedTable[-1][0])
        isTableEnd = self._is_table_end(tableName,mergedFields)
        if isTableEnd == True or len(tables) == 1:
            return processedTable, isTableEnd

        for index,table in enumerate(tables):
            table = [list(map(lambda x: str(x).replace('\n', ''), row)) for row in table]
            fieldList = [row[0] for row in table]
            mergedFields = reduce(self._merge, fieldList)
            # isTableEnd = self.self._is_table_end(tableName,table[-1][0])
            isTableEnd = self._is_table_end(tableName, mergedFields)
            isTableStart = self._is_table_start(tableName,table)
            if isTableStart == True:
                processedTable = table

            if isTableEnd == True:
                processedTable = table
                break
            else:
                #对于合并所所有者权益变动表,对某些情况下因为表尾字段做了拆分,很难通过表尾字段做判断,可以通过下一张表的开头来判断,上一张表的结束.
                if isTableStart == True:
                    if len(page_numbers) > 1 and index > 0:
                        processedTable = tables[index - 1]
                        isTableEnd = True
                        break

        return processedTable,isTableEnd

    def _is_table_start(self,tableName,table):
        #针对合并所有者权益表,第一个表头"项目",并不是出现在talbe[0][0],而是出现在第一列的第一个有效名称中
        isTableStart = False
        firstHeaderName = self.dictTables[tableName]['header'][0]
        headerList = [row[0] for row in table]
        for header in headerList:
            if self._is_valid(header):
                if firstHeaderName == header:
                    isTableStart = True
                break
        return isTableStart

    def _is_table_end(self,tableName,lastField):
        #对获取到的字段做标准化(需要的话),然后和配置表中代表最后一个字段(或模式)做匹配,如匹配到,则认为找到表尾
        #对于现金分红情况表,因为字段为时间,则用模式去匹配,匹配到一个即可认为找到表尾
        #针对合并所有者权益表,表尾字段"四、本期期末余额",并不是出现在talbe[-1][0],而是出现在第一列的最后两个字段,且有可能是分裂的
        isTableEnd = False
        fieldLast = self.dictTables[tableName]["fieldLast"]
        if isinstance(lastField,str) and isinstance(fieldLast,str) and fieldLast != '':
            matched = re.search(fieldLast,lastField)
            if matched is not None:
                isTableEnd = True
        return isTableEnd

    def _close(self):
        self._pdf.close()

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
