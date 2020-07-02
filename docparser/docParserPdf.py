#!/usr/bin/env Python
# coding=utf-8
# @Time    : 12/9/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserPdf.py
# @Note    : 用于年报,半年报,季报pdf文件的读写

from docparser.docParserBaseClass import *
import pdfplumber
import pandas as pd
from openpyxl import Workbook

class docParserPdf(docParserBase):
    def __init__(self,gConfig):
        super(docParserPdf,self).__init__(gConfig)
        self.interpretPrefix = ''
        self.table_settings = gConfig["table_settings"]
        self._load_data()

    def _load_data(self,input=None):
        self._pdf = pdfplumber.open(self.sourceFile,password='')
        self._data = self._pdf.pages
        self._index = 0
        self._length = len(self._data)

    def _get_text(self,page=None):
        #interpretPrefix用于处理比如 合并资产负债表分布在多个page页面的情况
        pageText = self.interpretPrefix + page.extract_text()
        return pageText

    def _get_tables(self,page = None):
        page = self.__getitem__(self._index-1)
        #page.extract_table()
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
        if dictTable is None:
            dictTable = dict()
        savedTable = dictTable['table']
        tableName = dictTable['tableName']
        fetchTables = self._get_tables()
        processedTable,isTableEnd = self._process_table(fetchTables,tableName)
        dictTable.update({'tableEnd':isTableEnd})
        if isinstance(savedTable, list):
            savedTable.extend(processedTable)
        else:
            savedTable = processedTable

        self.interpretPrefix = interpretPrefix
        if dictTable['tableBegin'] == True and dictTable['tableEnd'] == True:
            self.interpretPrefix = ''

        dictTable.update({'table':savedTable})
        return dictTable

    def _process_table(self,tables,tableName):
        lastFiledName = self.dictTables[tableName]['fieldName'][-1] #获取表的最后一个字段
        firstHeaderName = self.dictTables[tableName]['header'][0]
        #processedTable = [list(map(lambda x:str(x).replace('\n','').replace('None',''),row))
        #                  for row in tables[-1]]
        processedTable = [list(map(lambda x:str(x).replace('\n',''),row))
                          for row in tables[-1]]
        isTableEnd = (lastFiledName == processedTable[-1][0])
        if isTableEnd == True or len(tables) == 1:
            processedTable = processedTable
            return processedTable, isTableEnd

        for table in tables:
            #table = [list(map(lambda x: str(x).replace('\n','').replace('None',''),row)) for row in table]
            table = [list(map(lambda x: str(x).replace('\n', ''), row)) for row in table]
            isTableEnd = (lastFiledName == table[-1][0])
            isTableStart = (firstHeaderName == table[0][0])
            if isTableStart == True:
                processedTable = table

            if isTableEnd == True:
                processedTable = table
                break
        return processedTable,isTableEnd

    def _close(self):
        self._pdf.close()



    '''
    def parsePdfminer(self,sourceFile,targetFile):
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfpage import PDFDocument, PDFPage
        from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
        from pdfminer.converter import PDFPageAggregator
        from pdfminer.layout import LTTextBoxHorizontal, LAParams
        from pdfminer.pdfinterp import PDFTextState
        #解析PDF文本，并保存到TXT文件中
        sourceFile = os.path.join(self.data_directory,sourceFile)
        targetFile = os.path.join(self.working_directory,targetFile)
        targetFile = '.'.join([*targetFile.split('.')[:-1],'txt'])
        if os.path.exists(targetFile):
            os.remove(targetFile)
        fp = open(sourceFile, 'rb')
        # 用文件对象创建一个PDF文档分析器
        parser = PDFParser(fp)
        # 创建一个PDF文档
        doc = PDFDocument(parser)
        # 连接分析器，与文档对象
        parser.set_document(doc)
        #doc.set_parser(parser)

        # 提供初始化密码，如果没有密码，就创建一个空的字符串
        #doc.initialize()

        # 检测文档是否提供txt转换，不提供就忽略
        if not doc.is_extractable:
            raise PDFTextState.PDFTextExtractionNotAllowed
        else:
            # 创建PDF，资源管理器，来共享资源
            rsrcmgr = PDFResourceManager()
            # 创建一个PDF设备对象
            laparams = LAParams()
            device = PDFPageAggregator(rsrcmgr, laparams=laparams)
            # 创建一个PDF解释其对象
            interpreter = PDFPageInterpreter(rsrcmgr, device)

            # 循环遍历列表，每次处理一个page内容
            # doc.get_pages() 获取page列表
            #for page in doc.get_pages():
            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)
                # 接受该页面的LTPage对象
                layout = device.get_result()
                # 这里layout是一个LTPage对象 里面存放着 这个page解析出的各种对象
                # 一般包括LTTextBox, LTFigure, LTImage, LTTextBoxHorizontal 等等
                # 想要获取文本就获得对象的text属性，
                with open(targetFile, 'a') as f:
                    for x in layout:
                        if (isinstance(x, LTTextBoxHorizontal)):
                            results = x.get_text()
                            print(results)
                            f.write(results + "\n")

            outlines = doc.get_outlines()
            for (level, title, dest, a, se) in outlines:
                print(level, title)
    '''

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
    parser=docParserPdf(gConfig)
    parser.initialize()
    return parser
