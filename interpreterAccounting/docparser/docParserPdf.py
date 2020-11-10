#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 12/9/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserPdf.py
# @Note    : 用于年报,半年报,季报pdf文件的读写

from interpreterAccounting.docparser.docParserBaseClass import *
import pdfplumber
import csv
import itertools
import pandas as pd
from functools import reduce


class DocParserPdf(DocParserBase):
    def __init__(self,gConfig):
        super(DocParserPdf, self).__init__(gConfig)
        self._interpretPrefix = NULLSTR
        self.checkpointfilename = os.path.join(self.working_directory, gConfig['checkpointfile'])
        self.checkpointIsOn = self.gConfig['checkpointIsOn'.lower()]


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
        if self._index == 1:
            #解决贵州茅台年报中,贵州茅台酒股份有限公司2018 年年度报告,被解析成"贵州茅台酒股份有限公司 年年度报告 2018
            pageText = page.extract_text(y_tolerance=4)
        else:
            pageText = page.extract_text()
        if pageText is not None:
            pageText = self._interpretPrefix + pageText.rstrip() + self._get_tail()
        else:
            #千禾味业：2019年度审计报告.PDF文件中全部是图片,没有文字,需要做特殊处理
            pageText = self._interpretPrefix + self._get_tail()
            self.logger.error('the %s page %d\'s text of is be None' % (self.sourceFile,self._index))
        return pageText


    def _get_tail(self):
        #在每一页的结尾增加TAIL EOF
        #self._standardize(self.gJsonInterpreter['NAME'],self.gJsonInterpreter['TAIL'])为了解决（603960）克来机电：2017年年度报告.PDF
        #其(1) 现金流量表补充资料出现在页尾,如下"现金流量表补充资料　√适用 □不适用 审计报告第 68 页"
        tail = self._standardize(self.gJsonInterpreter['NAME'],self.gJsonInterpreter['TAIL']) + ' ' + EOF
        return tail


    def _get_tables(self,dictTable):
        if self._index == 0 or self._length == 0:
            # 当从interpreterAccountingUnitTest函数进入时,没有实际的pdf供解析,直接返回
            return
        page = self.__getitem__(self._index-1)
        table_settings = self._get_table_settings(dictTable)
        self._debug_extract_tables(page,table_settings)
        return page.extract_tables(table_settings=table_settings)


    def _get_table_settings(self,dictTable):
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
            else:
                value = str(value)
            return value
        DEFAULT_TABLE_SETTINGS = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "explicit_vertical_lines": [],
            "explicit_horizontal_lines": [],
            "snap_tolerance": DEFAULT_SNAP_TOLERANCE,
            "join_tolerance": DEFAULT_JOIN_TOLERANCE,
            "edge_min_length": 3,
            "min_words_vertical": DEFAULT_MIN_WORDS_VERTICAL,
            "min_words_horizontal": DEFAULT_MIN_WORDS_HORIZONTAL,
            "keep_blank_chars": False,
            "text_tolerance": 3,
            "text_x_tolerance": None,
            "text_y_tolerance": None,
            "intersection_tolerance": 3,
            "intersection_x_tolerance": None,
            "intersection_y_tolerance": None,
        }
        '''
        table_settings = self._get_special_settings(dictTable)
        return table_settings


    def _get_special_settings(self,dictTable):
        keyName = '默认值'
        table_settings = dict()
        snap_tolerance = self.gJsonBase['table_settings'][keyName]["snap_tolerance"]
        #新增join_tolerance,解决康泰生物：2018年年度报告.PDF中合并利润表,合并资产负债表的解析不正确问题
        join_tolerance = self.gJsonBase['table_settings'][keyName]["join_tolerance"]
        if dictTable['公司简称'] != NULLSTR:
            keyName = dictTable['公司简称']
            reportType = dictTable['报告类型']
            if keyName in self.gJsonBase['table_settings'].keys():
                #if dictTable['报告时间'] in self.gJsonBase['table_settings'][keyName]['报告时间'] \
                #    and dictTable['报告类型'] == self.gJsonBase['table_settings'][keyName]['报告类型']:
                if reportType in self.gJsonBase['table_settings'][keyName].keys():
                    if dictTable['报告时间'] in self.gJsonBase['table_settings'][keyName][reportType]:
                        snap_tolerance = self.gJsonBase['table_settings'][keyName]["snap_tolerance"]
                        if "join_tolerance" in self.gJsonBase['table_settings'][keyName].keys():
                            join_tolerance = self.gJsonBase['table_settings'][keyName]["join_tolerance"]
        table_settings.update({"snap_tolerance":snap_tolerance})
        table_settings.update({"join_tolerance":join_tolerance})
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
        processedTable,isTableEnd,isTableStart = self._process_table(page_numbers,fetchTables, tableName)
        dictTable.update({'tableEnd':isTableEnd})
        if len(page_numbers) == 1 and isTableStart == False and processedTable == NULLSTR:
            #这种情况下表明解释器在搜索时出现了误判,需要重置搜索条件,解决三诺生物2019年年报第60,61页出现了错误的合并资产负责表,真正的在第100页.
            #这种情况下还需要判断processedTable是否有效,如果有效,说明已经搜索到了,此时忽略isTableStart
            self.logger.warning('failed to fetchtable %s witch has invalid data,prefix : %s page %d!'
                                %(tableName,self.interpretPrefix.replace('\n',' '),page_numbers[0]))
            dictTable.update({'tableBegin': False})
            dictTable.update({'tableEnd': False})
            #解决尚荣医疗：2018年年度报告,合并所有者权益变动表在第二页搜索不到时,不再连续往下搜索,p106,107
            dictTable.update({'page_numbers':list()})
            self.interpretPrefix = NULLSTR
            return dictTable
        if isinstance(savedTable, list):
            savedTable.extend(processedTable)
        else:
            savedTable = processedTable
        if dictTable['tableBegin'] == True and dictTable['tableEnd'] == True:
            self.interpretPrefix = NULLSTR
        dictTable.update({'table':savedTable})
        return dictTable


    def _process_table(self,page_numbers,tables,tableName):
        processedTable,isTableEnd,isTableStart = NULLSTR , False,False
        assert isinstance(page_numbers,list) and len(page_numbers) > 0,"page_number(%s) must not be NULL"%page_numbers
        if len(tables) == 0:
            return processedTable,isTableEnd,isTableStart
        processedTable = [list(map(lambda x: str(x).replace('\n', NULLSTR), row)) for row in tables[-1]]
        processedTable = self._discard_last_row(processedTable,tableName)
        if len(processedTable) > 0:
            # 博通集成2019年年报, P93 ,搜到一张不需要的合并资产负债表,其中存在另外一张表只有一行, _discard_last_row处理后变成了空表
            #processedTable = NULLSTR
            #return processedTable, isTableEnd, isTableStart
            fieldList = [row[0] for row in processedTable]
            headerList = processedTable[0]
            #解决三诺生物2019年年报第60页,61页出现错误的合并资产负债表,需要跳过去
            isTableStart = False
            if len(processedTable[0]) > 1:
                secondFieldList = [row[1] for row in processedTable]
                isTableStart = self._is_table_start_simple(tableName, fieldList, secondFieldList,headerList)
            isTableEnd = self._is_table_end(tableName,fieldList)
        if len(tables) == 1:
            #（000652）泰达股份：2019年年度报告.PDF P40页出现了错误的普通股现金分红情况表的语句,这个时候不能够把带有值的processedTable返回
            if len(page_numbers) == 1 and isTableStart == False:
                self.logger.warning('failed to fetch %s whitch has invalid data:%s'%(tableName,processedTable))
                processedTable = NULLSTR
            return processedTable, isTableEnd,isTableStart
        processedTable = NULLSTR
        for index,table in enumerate(tables):
            table = [list(map(lambda x: str(x).replace('\n', NULLSTR), row)) for row in table ]
            table = self._discard_last_row(table,tableName)
            if len(table) == 0 or len(table[0]) <= 1:
                # 博通集成2019年年报, P93 ,搜到一张不需要的合并资产负债表,其中存在另外一张表只有一行, _discard_last_row处理后变成了空表
                #海螺水泥2015年年报，合并现金流量表数据解析错误，只有2列，第二列是附注，没有任何有效数据
                continue
            fieldList = [row[0] for row in table]
            if self._is_row_all_invalid(fieldList):
                self.logger.warning('the first row of tables is all invalid:%s'%table)
            secondFieldList = [row[1] for row in table]
            headerList = table[0]
            #浙江鼎力2018年年报,分季度主要财务数据,表头单独在一页中,而表头的第一个字段刚好为空,因此不能做mergedHeaders是否为空字符串的判断.
            isTableEnd = self._is_table_end(tableName, fieldList)
            isTableStart = self._is_table_start_simple(tableName, fieldList, secondFieldList,headerList)
            if len(page_numbers) == 1:
                #len(page_numers) == 1表示本表所在的第一页,需要明确判断出isTabletart = True 才能使得isTableEnd生效
                if isTableStart and isTableEnd:
                    processedTable = table
                    break
                elif isTableStart:
                    processedTable = table
                else:
                    #在第一页,没有搜索到表字段头的情况下搜索到了表尾,则是非法的: 荣盛发展2017年报P63 普通股现金分红情况表 出现了这种情况
                    isTableEnd = False
            else:
                if isTableEnd == True:
                    processedTable = table
                    #if isTableStart == False:
                        #解决（002812）恩捷股份：2018年年度报告.PDF,只能通过repair_list解决.主要会计数据分成两样,第二页出现一张统一控制下企业合并,和主要会计数据表字段完全一样,导致误判
                    break
                elif isTableStart == True:
                    processedTable = table
                else:
                    #正对华侨城A 2018年年报, 合并资产负债表 的 中间表出现在某一页,但是被拆成了两个表,需要被重新组合成一张新的表
                    if processedTable == NULLSTR:
                        processedTable = table
                    else:
                        processedTable.extend(table)
                        self.logger.warning('%s 的中间页出现的表被拆成多份,在此对表进行合并,just for debug!'%tableName)
        return processedTable,isTableEnd,isTableStart


    def _discard_last_row(self,table,tableName):
        #引入maxFieldLen是为了解决（002555）三七互娱：2018年年度报告.PDF,主要会计数据,在最后一个字段'归属于上市公司股东的净资产（元）'后面又加了一段无用的话,直接去掉
        maxFieldLen = self.dictTables[tableName]['maxFieldLen']
        if isinstance(table[-1][0],str) and len(table[-1][0]) > 2 * maxFieldLen:
            #去掉最后一个超长且无用的字段
            table = table[:-1]
        return  table


    def _is_table_start_simple(self,tableName,fieldList,secondFieldList,headerList):
        # 解决隆基股份2018年年度报告的无形资产情况,同一页中出现多张表也有相同的表头的第一字段'项目'
        # 针对合并所有者权益表,第一个表头"项目",并不是出现在talbe[0][0],而是出现在第一列的第一个有效名称中
        # 解决海螺水泥2018年年报中,主要会计数据的表头为'项 目'和规范的表头'主要会计数据'不一致,采用方法使得该表头失效
        # 解决通策医疗2019年年报中无形资产情况表所在的页中,存在另外一个表头 "项目名称",会导致用"^项目"去匹配时出现误判
        assert isinstance(fieldList, list) and isinstance(secondFieldList, list), \
            "fieldList and headerList must be list,but now get %s %s" % (type(fieldList), type(secondFieldList))
        isTableStart,isTableStartFirst,isTableStartSecond,isTableStartTree = False,False,False,False
        mergedFields = reduce(self._merge, fieldList)
        mergedFieldsSecond = reduce(self._merge, secondFieldList)
        mergedHeaders = reduce(self._merge, headerList)
        headerFirst = self.dictTables[tableName]["headerFirst"]
        headerSecond = self.dictTables[tableName]["headerSecond"]
        fieldFirst = self.dictTables[tableName]['fieldFirst']
        #assert fieldFirst != NULLSTR and headerFirst != NULLSTR and headerSecond != NULLSTR, 'the first field of %s must not be NULL' % tableName
        assert headerFirst != NULLSTR and headerSecond != NULLSTR, 'the first field of %s must not be NULL' % tableName
        #headerFirst,headerSecond,fieldFirst已经在_fields_replace_punctuate中把英文标点替换成中文了
        headerFirst = headerFirst.replace('(', '（').replace(')', '）').replace('[','（').replace(']','）')   # 在正则表达式中,'()'是元符号,需要替换成中文符号
        headerSecond = headerSecond.replace('(', '（').replace(')', '）').replace('[','（').replace(']','）')
        fieldFirst = fieldFirst.replace('(', '（').replace(')', '）').replace('[','（').replace(']','）')
        #考虑两种情况,表头的第一个字段为空,则直接以fieldFirst来匹配,如果不为空,则以表头第一个字段 + fieldFirst 来匹配
        patternHeaderFirst = '|'.join(['^' + header + field for (header,field)
                                       in itertools.product(headerFirst.split('|'),fieldFirst.split('|'))])
        patternHeaderSecond = '|'.join(['^' + field for field in headerSecond.split('|')])
        patternHeaders = '|'.join(['^' + header + headerNext for (header,headerNext)
                                   in itertools.product(headerFirst.split('|'),headerSecond.split('|'))])
        if isinstance(mergedFields, str) and isinstance(patternHeaderFirst, str) :
            mergedFields = mergedFields.replace('(', '（').replace(')', '）').replace('[','（').replace(']','）').replace(' ', NULLSTR)
            #mergedFields = self._replace_fieldname(mergedFields)
            matched = re.search(patternHeaderFirst, mergedFields)
            if matched is not None:
                isTableStartFirst = True
        if isinstance(mergedFieldsSecond, str) and isinstance(patternHeaderSecond, str) :
            mergedFieldsSecond = mergedFieldsSecond.replace('(', '（').replace(')', '）').replace('(', '（').replace(')', '）').replace('[','（').replace(']','）').replace(' ', NULLSTR)
            #mergedFieldsSecond = self._replace_fieldname(mergedFieldsSecond)
            matched = re.search(patternHeaderSecond, mergedFieldsSecond)
            if matched is not None:
                isTableStartSecond = True
        if isinstance(mergedHeaders,str) and isinstance(patternHeaders,str):
            #解决华东医药2015年年报,主营业务分行业经营情况, 第一行的第一列,第二列字段全部为空的场景
            mergedHeaders = mergedHeaders.replace('(', '（').replace(')', '）').replace('(', '（').replace(')', '）').replace('[','（').replace(']','）').replace(' ', NULLSTR)
            matched = re.search(patternHeaders,mergedHeaders)
            if matched is not None:
                isTableStartTree = True

        if tableName == '无形资产情况' :
            #解决杰瑞股份2018年年报无形资产情况,同一个页面出现了另外一张表,两张表的第一列完全相同,所以需要判断第二列结果才行
            #星源材质2019年年报中无形资产情况出现在页尾,且只有一行表头: 项目 土地使用权 专利权 非专利技术 软件及其他 合计. 这个时候isTableStartFirst失效
            #isTableStart = isTableStartFirst and isTableStartSecond
            isTableStart = isTableStartSecond
        elif tableName == '主营业务分行业经营情况':
            #解决宝来特2014年报,主营业务分行业经营情况表所在的页,出现两张第一列完全相同的表
            #解决九安医疗2014年财报,主营业务分行业经营情况 出现在页尾,且只有一行: 营业收入 营业成本 毛利率营业收入比上年同期增减营业成本比上年同期增减毛利率比上年同期
            #isTableStart = isTableStartFirst and isTableStartSecond
            isTableStart = isTableStartSecond or isTableStartTree
        #elif tableName == '主要会计数据':
            #解决安琪酵母2014年年报中同一页中还有 主要会计数据 和主要财务指标,结果误读了主要财务指标
        #    isTableStart = isTableStartFirst
        else:
            isTableStart = isTableStartFirst or isTableStartSecond
        return isTableStart


    def _is_table_end(self,tableName,fieldList):
        #对获取到的字段做标准化(需要的话),然后和配置表中代表最后一个字段(或模式)做匹配,如匹配到,则认为找到表尾
        #对于现金分红情况表,因为字段为时间,则用模式去匹配,匹配到一个即可认为找到表尾
        #针对合并所有者权益表,表尾字段"四、本期期末余额",并不是出现在talbe[-1][0],而是出现在第一列的最后两个字段,且有可能是分裂的
        assert isinstance(fieldList,list),"fieldList must be a list,bug now get %s"%type(fieldList)
        isTableEnd = False
        mergedFields = reduce(self._merge, fieldList)
        fieldLast = self.dictTables[tableName]["fieldLast"]
        #fieldLast 在self._fields_replace_punctuate中已经被替换过了
        #海康威视2014年报主要会计数据的最后一个字段为 归属于上市公司股东的每股净资产（元/股）[注],其中的 [] 会导致正则表达式匹配失败,需要替换
        fieldLast = fieldLast.replace('(','（').replace(')','）').replace('[','（').replace(']','）')  #在正则表达式中,'()'是元符号,需要替换成中文符号
        fieldLast = '|'.join([field + '$' for field in fieldLast.split('|')])
        if isinstance(mergedFields,str) and isinstance(fieldLast,str) and fieldLast != NULLSTR:
            mergedFields = mergedFields.replace('(','（').replace(')','）').replace('[','（').replace(']','）').replace(' ',NULLSTR).rstrip()
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


    def is_file_in_checkpoint(self,content):
        if self.checkpointIsOn == False:
            return False
        reader = self.get_checkpoint()
        if content in reader:
            return True


    def save_checkpoint(self,content):
        if self.checkpointIsOn == False:
            return
        with open(self.checkpointfilename, 'r+', newline='', encoding='utf-8') as checkpointfile:
            reader = checkpointfile.read().splitlines()
            reader = reader + [content]
            reader.sort()
            lines = [line + '\n' for line in reader]
            checkpointfile.seek(0)
            checkpointfile.truncate()
            checkpointfile.writelines(lines)


    def get_checkpoint(self):
        if self.checkpointIsOn == False:
            return
        with open(self.checkpointfilename, 'r', encoding='utf-8') as csv_in:
            reader = csv_in.read().splitlines()
        return reader


    def remove_checkpoint_files(self,sourcefiles):
        assert isinstance(sourcefiles,list),'Parameter sourcefiles must be list!'
        if self.checkpointIsOn == False:
            return

        with open(self.checkpointfilename, 'r+', newline='', encoding='utf-8') as checkpointfile:
            reader = checkpointfile.read().splitlines()
            resultfiles = list(set(reader).difference(set(sourcefiles)))
            resultfiles.sort()
            lines = [line + '\n' for line in resultfiles]
            checkpointfile.seek(0)
            checkpointfile.truncate()
            checkpointfile.writelines(lines)
            removedfiles = list(set(reader).difference(set(resultfiles)))
        if len(removedfiles) > 0:
            removedlines = '\n\t\t\t\t'.join(removedfiles)
            self.logger.info("Success to remove from checkpointfile : %s"%(removedlines))


    @property
    def interpretPrefix(self):
        return self._interpretPrefix


    @interpretPrefix.setter
    def interpretPrefix(self,prefix):
        assert isinstance(prefix,str),"para(%s) of set_interpretPrefix must be string"%prefix
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
        if self.checkpointIsOn:
            if not os.path.exists(self.checkpointfilename):
                fw = open(self.checkpointfilename,'w',newline='',encoding='utf-8')
                fw.close()
        else:
            if os.path.exists(self.checkpointfilename):
                os.remove(self.checkpointfilename)


def create_object(gConfig):
    parser=DocParserPdf(gConfig)
    parser.initialize()
    return parser
