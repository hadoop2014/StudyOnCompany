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
        self.model_savefile = os.path.join(self.working_directory,self._get_class_name(self.gConfig) + '.model')
        self.source_directory = os.path.join(self.data_directory,self.gConfig['source_directory'])
        self.sourceFile = os.path.join(self.data_directory,self.gConfig['source_directory'],self.gConfig['sourcefile'])
        self.taskResult = os.path.join(self.gConfig['working_directory'],self.gConfig['taskResult'.lower()])


    def _get_standardized_header(self,headerList,tableName):
        assert headerList is not None, 'sourceRow(%s) must not be None' % headerList
        fieldStandardize = self.dictTables[tableName]['headerStandardize']
        if isinstance(headerList, list):
            standardizedFields = [self._standardize(fieldStandardize, field) for field in headerList]
        else:
            standardizedFields = self._standardize(fieldStandardize, headerList)
        return standardizedFields


    def _is_header_in_row(self,row,tableName):
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


    def _get_class_name(self, gConfig):
        parser_name = re.findall('DocParser(.*)', self.__class__.__name__).pop().lower()
        assert parser_name in gConfig['docformatlist'], \
            'docformatlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['docformatlist'], parser_name, self.__class__.__name__)
        return parser_name


    def saveCheckpoint(self):
        pass


    def getSaveFile(self):
        if self.model_savefile == NULLSTR:
            self.model_savefile = None
            return None
        if self.model_savefile is not None:
            if os.path.exists(self.model_savefile) == False:
                return None
                # 文件不存在
        return self.model_savefile


    def removeSaveFile(self):
        if self.model_savefile is not None:
            filename = os.path.join(os.getcwd(), self.model_savefile)
            if os.path.exists(filename):
                os.remove(filename)


    def debug_info(self, info=None):
        if self.debugIsOn == False:
            return
        pass
        return


    def debug(self, layer, name=NULLSTR):
        pass


    def clear_logging_directory(self,logging_directory):
        assert logging_directory == self.logging_directory ,\
            'It is only clear logging directory, but %s is not'%logging_directory
        files = os.listdir(logging_directory)
        for file in files:
            full_file = os.path.join(logging_directory,file)
            if os.path.isdir(full_file):
                self.clear_logging_directory(full_file)
            else:
                try:
                    os.remove(full_file)
                except:
                   print('%s is not be removed'%full_file)


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
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        self.clear_logging_directory(self.logging_directory)