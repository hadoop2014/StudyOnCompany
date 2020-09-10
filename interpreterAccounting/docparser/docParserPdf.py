#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 12/9/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserPdf.py
# @Note    : 用于年报,半年报,季报pdf文件的读写

from interpreterAccounting.docparser.docParserBaseClass import *
import pdfplumber
from functools import reduce

class DocParserPdf(DocParserBase):
    def __init__(self,gConfig):
        super(DocParserPdf, self).__init__(gConfig)
        self._interpretPrefix = NULLSTR
        self.table_settings = gConfig["table_settings"]
        #self._load_data()

    def _load_data(self,sourceFile=None):
        if sourceFile is not None and sourceFile != NULLSTR:
            self.sourceFile = os.path.join(self.data_directory,self.gConfig['source_directory'],sourceFile)
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
        self._debug_extract_tables(page,table_settings)
        return page.extract_tables(table_settings=table_settings)

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
        snap_tolerance = self.gJsonBase['table_settings'][keyName]["snap_tolerance"]
        if dictTable['公司简称'] != NULLSTR:
            keyName = dictTable['公司简称']
            if keyName in self.gJsonBase['table_settings'].keys():
                if dictTable['报告时间'] == self.gJsonBase['table_settings'][keyName]['报告时间'] \
                    and dictTable['报告类型'] == self.gJsonBase['table_settings'][keyName]['报告类型']:
                    snap_tolerance = self.gJsonBase['table_settings'][keyName]["snap_tolerance"]
        table_settings.update({"snap_tolerance":snap_tolerance})
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
        processedTable,isTableEnd = self._process_table(page_numbers,fetchTables, tableName)
        dictTable.update({'tableEnd':isTableEnd})
        if isinstance(savedTable, list):
            savedTable.extend(processedTable)
        else:
            savedTable = processedTable
        if dictTable['tableBegin'] == True and dictTable['tableEnd'] == True:
            self.interpretPrefix = NULLSTR
        dictTable.update({'table':savedTable})
        return dictTable

    def _process_table(self,page_numbers,tables,tableName):
        processedTable,isTableEnd = NULLSTR , False
        assert isinstance(page_numbers,list) and len(page_numbers) > 0,"page_number(%s) must not be NULL"%page_numbers
        if len(tables) == 0:
            return processedTable,isTableEnd
        processedTable = [list(map(lambda x: str(x).replace('\n', NULLSTR), row)) for row in tables[-1]]
        fieldList = [row[0] for row in processedTable]
        isTableEnd = self._is_table_end(tableName,fieldList)
        if len(tables) == 1:
            return processedTable, isTableEnd

        processedTable = NULLSTR
        for index,table in enumerate(tables):
            table = [list(map(lambda x: str(x).replace('\n', NULLSTR), row)) for row in table]
            if len(table[0]) <= 1:
                continue
            fieldList = [row[0] for row in table]
            secondFieldList = [row[1] for row in table]
            headerList = table[0]
            #浙江鼎力2018年年报,分季度主要财务数据,表头单独在一页中,而表头的第一个字段刚好为空,因此不能做mergedHeaders是否为空字符串的判断.
            isTableEnd = self._is_table_end(tableName, fieldList)
            #isTableStart = self._is_table_start(tableName,fieldList,headerList)
            isTableStart = self._is_table_start_simple(tableName, fieldList, secondFieldList)
            if isTableStart == True:
                processedTable = table
            if len(page_numbers) == 1:
                #len(page_numers) == 1表示本表所在的第一页,需要明确判断出isTabletart = True 才能使得isTableEnd生效
                if isTableStart and isTableEnd:
                    processedTable = table
                    break
            elif isTableEnd == True :
                processedTable = table
                break
        return processedTable,isTableEnd

    def _is_table_start_simple(self,tableName,fieldList,secondFieldList):
        assert isinstance(fieldList, list) and isinstance(secondFieldList, list), \
            "fieldList and headerList must be list,but now get %s %s" % (type(fieldList), type(secondFieldList))
        isTableStart = False
        mergedFields = reduce(self._merge, fieldList)
        mergedFieldsSecond = reduce(self._merge, secondFieldList)
        headerFirst = self.dictTables[tableName]["header"][0]
        headerSecond = self.dictTables[tableName]["header"][1]
        fieldFirst = self.dictTables[tableName]['fieldFirst']
        assert fieldFirst != NULLSTR and headerFirst != NULLSTR and headerSecond != NULLSTR, 'the first field of %s must not be NULL' % tableName
        headerFirst = headerFirst.replace('(', '（').replace(')', '）')  # 在正则表达式中,'()'是元符号,需要替换成中文符号
        headerSecond = headerSecond.replace('(', '（').replace(')', '）')
        fieldFirst = fieldFirst.replace('(', '（').replace(')', '）')
        #考虑两种情况,表头的第一个字段为空,则直接以fieldFirst来匹配,如果不为空,则以表头第一个字段 + fieldFirst 来匹配
        patternHeaderFirst = '|'.join(['^' + field for field in fieldFirst.split('|')]
                                    +['^' + headerFirst + field for field in fieldFirst.split('|')])
        patternHeaderSecond = '^' + headerSecond
        if isinstance(mergedFields, str) and isinstance(patternHeaderFirst, str) :
            mergedFields = mergedFields.replace('(', '（').replace(')', '）').replace(' ', NULLSTR)
            matched = re.search(patternHeaderFirst, mergedFields)
            if matched is not None:
                isTableStart = True
        if isinstance(mergedFieldsSecond, str) and isinstance(patternHeaderSecond, str) :
            mergedFieldsSecond = mergedFieldsSecond.replace('(', '（').replace(')', '）').replace(' ', NULLSTR)
            matched = re.search(patternHeaderSecond, mergedFieldsSecond)
            if matched is not None:
                isTableStart = True
        return isTableStart

    def _is_table_start(self,tableName,fieldList,headerList):
        #针对合并所有者权益表,第一个表头"项目",并不是出现在talbe[0][0],而是出现在第一列的第一个有效名称中
        assert isinstance(fieldList,list) and isinstance(headerList,list),\
            "fieldList and headerList must be list,but now get %s %s"%(type(fieldList),type(headerList))
        isTableStart = False
        mergedFields = reduce(self._merge, fieldList)
        mergedHeaders = reduce(self._merge, headerList)
        firstHeaderInRow = headerList[0]
        headerFirst = self.dictTables[tableName]["header"][0]
        if headerFirst ==NULLSTR:
            headerFirst = self._get_standardized_header(self.dictTables[tableName]['header'][1], tableName)
        fieldFirst = self.dictTables[tableName]['fieldFirst']
        assert fieldFirst != NULLSTR,'the first field of %s must not be NULL'%tableName
        if headerFirst == NULLSTR or firstHeaderInRow == NULLSTR:
            #headerFirst == NULLSTR针对分季度主要财务数据的场景
            #firstHaderInRow == NULLSTR针对主要会计数据中部分财报第一个字段为空(本应该为'主要会计数据')
            if firstHeaderInRow == NULLSTR:
                #headerFirstTemp = self._get_standardized_header(self.dictTables[tableName]['header'][1], tableName)
                headerFirstTemp = self.dictTables[tableName]['header'][1]
            else:
                headerFirstTemp = headerFirst
            headerFirstTemp = headerFirstTemp.replace('(', '（').replace(')', '）')
            assert headerFirstTemp != NULLSTR,'the second header of %s must not be NULL'%tableName
            if isinstance(mergedHeaders, str) and isinstance(headerFirstTemp, str):
                mergedHeaders = mergedHeaders.replace('(', '（').replace(')', '）').replace(' ',NULLSTR)
                matched = re.search('^' + headerFirstTemp, mergedHeaders)
                if matched is not None:
                    isTableStart = True
        elif headerFirst != firstHeaderInRow and self._is_header_in_row(headerList,tableName):
            #解决海螺水泥2018年年报中,主要会计数据的表头为'项 目'和规范的表头'主要会计数据'不一致,采用方法使得该表头失效
            fieldFirst = firstHeaderInRow.replace(' ',NULLSTR) + fieldFirst
            #解决通策医疗2019年年报中无形资产情况表所在的页中,存在另外一个表头 "项目名称",会导致用"^项目"去匹配时出现误判
            headerFirst = NULLSTR
        headerFirst = headerFirst.replace('(', '（').replace(')', '）')  # 在正则表达式中,'()'是元符号,需要替换成中文符号
        fieldFirst = fieldFirst.replace('(', '（').replace(')', '）')
        if headerFirst != NULLSTR:
            #fieldFirst = '^' + fieldFirst
            headerFirst = '^' + headerFirst
            #headerFirst = '|'.join([headerFirst,fieldFirst])
            #解决隆基股份2018年年度报告的无形资产情况,同一页中出现多张表也有相同的表头的第一字段'项目',
            headerFirst = ''.join([headerFirst, fieldFirst])
        else:
            fieldFirst = '^' + fieldFirst
            headerFirst = fieldFirst
        if isinstance(mergedFields, str) and isinstance(headerFirst, str) and headerFirst != NULLSTR:
            mergedFields = mergedFields.replace('(', '（').replace(')', '）').replace(' ',NULLSTR)
            matched = re.search(headerFirst, mergedFields)
            if matched is not None:
                isTableStart = True
        return isTableStart

    def _is_table_end(self,tableName,fieldList):
        #对获取到的字段做标准化(需要的话),然后和配置表中代表最后一个字段(或模式)做匹配,如匹配到,则认为找到表尾
        #对于现金分红情况表,因为字段为时间,则用模式去匹配,匹配到一个即可认为找到表尾
        #针对合并所有者权益表,表尾字段"四、本期期末余额",并不是出现在talbe[-1][0],而是出现在第一列的最后两个字段,且有可能是分裂的
        assert isinstance(fieldList,list),"fieldList must be a list,bug now get %s"%type(fieldList)
        isTableEnd = False
        mergedFields = reduce(self._merge, fieldList)
        fieldLast = self.dictTables[tableName]["fieldLast"]
        fieldLast = fieldLast.replace('(','（').replace(')','）')  #在正则表达式中,'()'是元符号,需要替换成中文符号
        fieldLast = '|'.join([field + '$' for field in fieldLast.split('|')])
        if isinstance(mergedFields,str) and isinstance(fieldLast,str) and fieldLast != NULLSTR:
            mergedFields = mergedFields.replace('(','（').replace(')','）').replace(' ',NULLSTR)
            matched = re.search(fieldLast,mergedFields)
            if matched is not None:
                isTableEnd = True
        return isTableEnd

    def _close(self):
        self._pdf.close()

    def _debug_extract_tables(self,page,table_settings):
        if self.debugIsOn == False:
            return
        image = page.to_image()
        image.reset().debug_tablefinder(table_settings)
        tables = page.extract_tables(table_settings)
        #image.draw_rects(tables)
        for table in tables:
            for row in table:
                self.logger.info('debug:' + str(row))

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
