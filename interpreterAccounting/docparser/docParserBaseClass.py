#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/25/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserBaseClass.py
import time
from interpreterAccounting.interpreterBaseClass import *
from functools import reduce

#文档解析模块的基类
class DocParserBase(InterpreterBase):
    def __init__(self,gConfig):
        super(DocParserBase, self).__init__(gConfig)
        self.start_time = time.time()
        #self.model_savefile = os.path.join(self.workingspace.directory,self._get_class_name(self.gConfig) + '.model')
        #self.source_directory = os.path.join(self.data_directory,self.gConfig['source_directory'])
        self.sourceFile = os.path.join(self.data_directory,self.gConfig['source_directory'],self.gConfig['sourcefile'])

    def _get_standardized_header(self,headerList,tableName):
        assert headerList is not None, 'sourceRow(%s) must not be None' % headerList
        fieldStandardize = self.dictTables[tableName]['headerStandardize']
        if isinstance(headerList, list):
            standardizedFields = [self.standard._standardize(fieldStandardize, field) for field in headerList]
        else:
            standardizedFields = self.standard._standardize(fieldStandardize, headerList)
        return standardizedFields


    def _is_header_in_row(self,row,tableName):
        """
        args:
            row - dataFrame数据中的一行
            tableName - 表名, 用于从interpreterAccounting.Json中获取参数的索引
        return:
            isHeaderInRow - True or False, True 表示在本行中有表头字段. 判断原则:
            1) 如果 row不是表格,且少于一个字段,返回 False
            2) 如果地一个字段不为空,且第一个字段在firstHeader配置中,则返回True
            3) 如果是转置表(isHorizontalTable=True),则用fieldStandardize,否则用headerStandardize对row中内容进行检索,检索到了返回True
        """
        isHeaderInRow = False
        if isinstance(row,list) == False:
            return isHeaderInRow
        elif len(row) <= 1:
            return isHeaderInRow
        firstHeader = self.dictTables[tableName]['headerFirst'].split('|')
        firstHeaderInRow = row[0]
        if firstHeaderInRow != NULLSTR and firstHeaderInRow in firstHeader:
            #解决中顺洁柔2019年报中,标题行出现"项目, None, None, None, None, None, None, None, None"的场景
            isHeaderInRow = True
            return isHeaderInRow
        mergedRow = reduce(self._merge, row[1:])
        isHorizontalTable = self.dictTables[tableName]['horizontalTable']
        if isHorizontalTable:
            #对于普通股现金分红情况表,表头标准化等于字段标准化
            headerStandardize = self.dictTables[tableName]['fieldStandardize']
        else:
            headerStandardize = self.dictTables[tableName]['headerStandardize']
        isHeaderInRow = self._is_field_matched(headerStandardize, mergedRow)
        return isHeaderInRow


    def _is_field_matched(self,pattern,field):
        isFieldMatched = False
        if isinstance(pattern, str) and isinstance(field, str):
            if pattern != NULLSTR:
                matched = re.search(pattern,field)
                if matched is not None:
                    isFieldMatched = True
        return isFieldMatched


    def _is_row_all_invalid(self,row):
        mergedField = reduce(self._merge,row)
        #解决上峰水泥2017年中出现" ,None,None,None,None,None"的情况,以及其他年报中出现"None,,None,,None"的情况.
        isRowAllInvalid = not self._is_valid(mergedField)
        return isRowAllInvalid


    def debug_info(self, info=None):
        if self.debugIsOn == False:
            return
        pass
        return


    def debug(self, layer, name=NULLSTR):
        pass

    # 装饰器，用于在unittest模式下，只返回一个数据，快速迭代
    @staticmethod
    def getdataForUnittest(getdata):
        def wapper(self, batch_size):
            if self.unitestIsOn == True:
                # 仅用于unitest测试程序
                def reader():
                    for (X, y) in getdata(self,batch_size):
                        yield (X, y)
                        break
                return reader()
            else:
                return getdata(self,batch_size)
        return wapper


    @getdataForUnittest.__get__(object)
    def getTrainData(self,batch_size):
        return


    def initialize(self):
        self.loggingspace.clear_directory(self.loggingspace.directory)