#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 12/9/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserPdf.py
# @Note    : 用于年报,半年报,季报pdf文件的读写

from interpreterAccounting.docparser.docParserBaseClass import *
import pdfplumber
import itertools
from functools import reduce


class DocParserPdf(DocParserBase):
    def __init__(self,gConfig):
        super(DocParserPdf, self).__init__(gConfig)
        self._interpretPrefix = NULLSTR
        self.debugExtractTable = self.gConfig["debugExtractTable".lower()]


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
        pageText = None
        try:
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
        except Exception as e:
            # 解决 （601139）深圳燃气：2015年半年度报告.PDF 解析时出现异常
            self.logger.error('some error occured in function page.extract_text:%s'%str(e))
        return pageText


    def _get_tail(self):
        #在每一页的结尾增加TAIL EOF
        #self.standard._standardize(self.gJsonInterpreter['NAME'],self.gJsonInterpreter['TAIL'])为了解决（603960）克来机电：2017年年度报告.PDF
        #其(1) 现金流量表补充资料出现在页尾,如下"现金流量表补充资料　√适用 □不适用 审计报告第 68 页"
        tail = self.standard._standardize(self.gJsonInterpreter['NAME'],self.gJsonInterpreter['TAIL']) + ' ' + EOF
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
            keyName = self.standard._get_company_alias(dictTable['公司简称'])
            reportType = dictTable['报告类型']
            if keyName in self.gJsonBase['table_settings'].keys():
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

        processedTable, isTableEnd, tableStartScore, isFirstRowAllInvalid = self._process_table(page_numbers, fetchTables, tableName)
        dictTable.update({'tableEnd':isTableEnd})
        dictTable.update({'firstRowAllInvalid': isFirstRowAllInvalid})
        if len(page_numbers) == 1 and tableStartScore == 0 and processedTable == NULLSTR:
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
            if dictTable['tableEnd'] == False and \
                (tableStartScore > dictTable['tableStartScore'] or
                (tableStartScore == dictTable['tableStartScore'] and page_numbers[-1] - page_numbers[-2] > 1)):
                # 如果第二次搜索到同一张表, 而且tableStartScore更高, 且isTableEnd = False,则认为是正确的表, 则重新开始拼接表
                # 解决国城矿业：2019年年度报告, 合并资产负债表出现两次, 后面一次是正确的
                # 华侨城A：2020年半年度报告, 合并所有者权益变动表 P83页第一次出现 tableStartScore = 2, 但是P85页再次出现的表 tableStartScore = 3
                #（600109）国金证券：2020年半年度报告.PDF,P43页第一次出现合并资产负债表,P57页出现真正的合并资产负债表
                # 海天味业：2016年年度报告,合并资产负债表,每一个分页上都有合并资产负债表的表头,导致tableStartScore >= last one成立,增加page_numbers[-1] - page_numbers[-2] > 1的判断
                savedTable = processedTable
                self.logger.info('second searched table %s at page %d, whitch tableStartScore(%d) >= last one(%d)'
                                 %(tableName, page_numbers[-1], tableStartScore, dictTable['tableStartScore']))
                dictTable.update({'page_numbers':[page_numbers[-1]]})
                dictTable.update({'tableStartScore': tableStartScore})
            else:
                savedTable.extend(processedTable)
        else:
            savedTable = processedTable
            dictTable.update({'tableStartScore': tableStartScore})
        if dictTable['tableBegin'] == True and dictTable['tableEnd'] == True:
            self.interpretPrefix = NULLSTR
        dictTable.update({'table':savedTable})
        return dictTable


    def _process_table(self,page_numbers,tables,tableName):
        processedTable, isTableEnd, tableStartScore, isFirstRowAllInvalid = NULLSTR , False, 0, False
        assert isinstance(page_numbers,list) and len(page_numbers) > 0,"page_number(%s) must not be NULL"%page_numbers
        if tables is None or len(tables) == 0:
            #tables = None是从interpreterAccountingUnittest.py调用时出现的情景
            return processedTable, isTableEnd, tableStartScore,isFirstRowAllInvalid
        processedTable = [list(map(lambda x: str(x).replace('\n', NULLSTR), row)) for row in tables[-1]]
        processedTable = self._discard_last_row(processedTable,tableName)
        if len(processedTable) > 0:
            # 博通集成2019年年报, P93 ,搜到一张不需要的合并资产负债表,其中存在另外一张表只有一行, _discard_last_row处理后变成了空表
            fieldList = [row[0] for row in processedTable]
            if self._is_row_all_invalid(fieldList) :
                isFirstRowAllInvalid = True
                self.logger.warning('the first row of tables is all invalid:%s'% processedTable)
            #解决三诺生物2019年年报第60页,61页出现错误的合并资产负债表,需要跳过去
            tableStartScore = self._is_table_start(tableName, fieldList, processedTable)
            isTableEnd = self._is_table_end(tableName,fieldList)
        if len(tables) == 1:
            #（000652）泰达股份：2019年年度报告.PDF P40页出现了错误的普通股现金分红情况表的语句,这个时候不能够把带有值的processedTable返回
            if len(page_numbers) == 1 and tableStartScore == 0:
                self.logger.warning('failed to fetch %s whitch has invalid data:%s'%(tableName,processedTable))
                processedTable = NULLSTR
            return processedTable, isTableEnd, tableStartScore, isFirstRowAllInvalid
        processedTable = NULLSTR
        # 引入maxTableStart 解决鲁商发展 2020年半年报P12页主营业务分行业经营情况表 出现了两张表头几乎一样的表,使得之前的isTableStart判断失效,改为置信度算法
        maxTableStartScore = 0
        maxTableEnd = False
        maxFirstRowAllInvalid = False
        for index,table in enumerate(tables):
            isFirstRowAllInvalid = False
            table = [list(map(lambda x: str(x).replace('\n', NULLSTR), row)) for row in table ]
            table = self._discard_last_row(table,tableName)
            if len(table) == 0 or len(table[0]) <= 1:
                # 博通集成2019年年报, P93 ,搜到一张不需要的合并资产负债表,其中存在另外一张表只有一行, _discard_last_row处理后变成了空表
                #海螺水泥2015年年报，合并现金流量表数据解析错误，只有2列，第二列是附注，没有任何有效数据
                continue
            fieldList = [row[0] for row in table]
            if self._is_row_all_invalid(fieldList):
                isFirstRowAllInvalid = True
                self.logger.warning('the first row of tables is all invalid:%s'%table)
            #浙江鼎力2018年年报,分季度主要财务数据,表头单独在一页中,而表头的第一个字段刚好为空,因此不能做mergedHeaders是否为空字符串的判断.
            tableStartScore = self._is_table_start(tableName, fieldList, table)
            isTableEnd = self._is_table_end(tableName, fieldList)
            if len(page_numbers) == 1:
                #len(page_numers) == 1表示本表所在的第一页,需要明确判断出isTabletart = True 才能使得isTableEnd生效
                if (isTableEnd and tableStartScore > 0 and maxTableEnd == False) \
                    or (isTableEnd and not maxTableEnd and tableStartScore >= maxTableStartScore and maxTableStartScore > 0)\
                    or (isTableEnd and maxTableEnd and tableStartScore > maxTableStartScore and maxTableStartScore > 0):
                    # (isTableEnd and not maxTableEnd and tableStartScore >= maxTableStartScore and maxTableStartScore > 0)
                    # 解决片仔癀：2016年年度报告,分季度财务数据的解析问题,出现两张表,应该取后面一张
                    # and maxTableEnd == False 解决（601878）浙商证券：2020年半年度报告,主要会计数据的解析问题
                    # (isTableEnd and maxTableEnd and tableStartScore > maxTableStartScore and maxTableStartScore > 0)
                    # 国检集团：2019年年度报告, 主要会计数据同一页出现两张表,应该取前面一张
                    processedTable = table
                    maxTableStartScore = tableStartScore
                    maxTableEnd = isTableEnd
                    maxFirstRowAllInvalid = isFirstRowAllInvalid
                    #break
                #elif tableStartScore > maxTableStartScore and maxTableEnd:
                #    ...
                    #（002407）多氟多：2020年年度报告,主要会计解析不对, 出现两张表,第一张表 tableStartScore = 2, isTableEnd = True为正确的表
                    # 第二张表 tableStartScore = 3, isTableEnd = False为错误的表.
                    # 这种情况 do nothing,, 通过 "headerSecond": "\\d+\\s*年|本报告期末|年初至报告期末|本报告期|本年比上", 来解决
                    # （300184）力源信息：2020年年度报告, 主营业务分行业经营情况, 第一张表 tableStartScore = 1, isTableEnd = True为错误的表,
                    #  第二张tableStartScore = 2, isTableEnd = False为正确的表,  必须去掉该分支
                elif tableStartScore > maxTableStartScore:
                    processedTable = table
                    maxTableStartScore = tableStartScore
                    maxTableEnd = isTableEnd
                    maxFirstRowAllInvalid = isFirstRowAllInvalid
                else:
                    #在第一页,没有搜索到表字段头的情况下搜索到了表尾,则是非法的: 荣盛发展2017年报P63 普通股现金分红情况表 出现了这种情况
                    #if maxTableStart == 0:
                    #    # 而且之前没有搜索到表头的情况下设置isTableEnd = False
                    isTableEnd = False
            else:
                if isTableEnd == True:
                    processedTable = table
                    maxTableEnd = isTableEnd
                    maxFirstRowAllInvalid = isFirstRowAllInvalid
                    #if isTableStart == False:
                        #解决（002812）恩捷股份：2018年年度报告.PDF,只能通过repair_list解决.主要会计数据分成两样,第二页出现一张统一控制下企业合并,和主要会计数据表字段完全一样,导致误判
                    break
                elif tableStartScore > maxTableStartScore:
                    processedTable = table
                    maxTableStartScore = tableStartScore
                    maxFirstRowAllInvalid = isFirstRowAllInvalid
                else:
                    #正对华侨城A 2018年年报, 合并资产负债表 的 中间表出现在某一页,但是被拆成了两个表,需要被重新组合成一张新的表
                    if processedTable == NULLSTR:
                        processedTable = table
                    else:
                        processedTable.extend(table)
                        self.logger.warning('%s 的中间页出现的表被拆成多份,在此对表进行合并,just for debug!'%tableName)
        return processedTable, maxTableEnd, maxTableStartScore, maxFirstRowAllInvalid


    def _discard_last_row(self,table,tableName):
        #引入maxFieldLen是为了解决（002555）三七互娱：2018年年度报告.PDF,主要会计数据,在最后一个字段'归属于上市公司股东的净资产（元）'后面又加了一段无用的话,直接去掉
        maxFieldLen = self.dictTables[tableName]['maxFieldLen']
        maxFieldLenPlus = self.dictTables[tableName]['maxFieldLenPlus']
        # 广济药业：2019年半年度报告,合并现金流量表,最后一行为 法定代表人：安靖 主管会计工作负责人:胡明峰 会计机构负责人：王琼,长度为 33,因此把此处从2改为1.3
        # 避免 光明乳业：2015年年度报告,无形资产情况, 最后一个字段出现误判
        #table[-1][0].replace(' ',NULLSTR)增加了replace(' ',NULLSTR)解决赛轮轮胎：2020年年度报告,无形资产情况表,最后一个字段增加了几个空格,导致if语句误判
        #if isinstance(table[-1][0],str) and len(table[-1][0].replace(' ',NULLSTR)) > 1.3 * maxFieldLen:
        if isinstance(table[-1][0], str) and len(table[-1][0].replace(' ', NULLSTR)) > maxFieldLen + maxFieldLenPlus:
            #去掉最后一个超长且无用的字段
            table = table[:-1]
        if len(table) > 0:
            #if isinstance(table[0][0],str) and len(table[0][0].replace(' ',NULLSTR)) > 1.5 * maxFieldLen:
            if isinstance(table[0][0], str) and len(table[0][0].replace(' ', NULLSTR)) > maxFieldLen + maxFieldLenPlus:
            #解决广济药业2015年报,合并资产负债表的第一个单元格为: 合并资产负债表 编制单位：湖北广济药业股份有限公司 2015年12月31日 单位：人民币元'
                table = table[1:]
        return  table


    def _is_table_start(self,tableName,fieldList,table):
        '''
        explain: 通过匹配表头字段判断是否是目标表的开始
        args:
            tableName - 目标表名称
            fieldList - 目标表第一列,即字段所在的列
            table - 目标表内容
        return:
            tableStartScore - 表头字段的匹配分数,分数越高就越可能是目标表
            1) 匹配第一列,即第一个表头 + 字段列
            2) 匹配第二列,即第二个表头
            3) 匹配第三列,即第三个表头
            4) 匹配第一行,即第一表头 + 第二表头 + 第三表头
        '''
        headerList = table[0]
        # 解决三诺生物2019年年报第60页,61页出现错误的合并资产负债表,需要跳过去
        tableStartScore = 0
        thirdFieldList = NULLSTR
        if len(table[0]) > 1:
            secondFieldList = [row[1] for row in table]
            if len(table[0]) > 2:
                thirdFieldList = [row[2] for row in table]
            tableStartScore = self._is_table_start_simple(tableName, fieldList, secondFieldList, thirdFieldList,
                                                          headerList)
        return tableStartScore


    #@pysnooper.snoop()
    def _is_table_start_simple(self,tableName,fieldList,secondFieldList,thirdFieldList,headerList):
        # 解决隆基股份2018年年度报告的无形资产情况,同一页中出现多张表也有相同的表头的第一字段'项目'
        # 针对合并所有者权益表,第一个表头"项目",并不是出现在talbe[0][0],而是出现在第一列的第一个有效名称中
        # 解决海螺水泥2018年年报中,主要会计数据的表头为'项 目'和规范的表头'主要会计数据'不一致,采用方法使得该表头失效
        # 解决通策医疗2019年年报中无形资产情况表所在的页中,存在另外一个表头 "项目名称",会导致用"^项目"去匹配时出现误判
        assert isinstance(fieldList, list) and isinstance(secondFieldList, list), \
            "fieldList and headerList must be list,but now get %s %s" % (type(fieldList), type(secondFieldList))
        mergedFields = reduce(self._merge, fieldList)
        #考虑两种情况,表头的第一个字段为空,则直接以fieldFirst来匹配,如果不为空,则以表头第一个字段 + fieldFirst 来匹配
        patternHeaderFirst = self._get_table_header_pattern(tableName,'headerFirst','fieldFirst')
        isTableStartFirst = self._table_start_match(mergedFields,patternHeaderFirst)

        #解决华东医药2015年年报,主营业务分行业经营情况, 第一行的第一列,第二列字段全部为空的场景,采用repair_list修复,不再采用这个
        #解决鲁商发展2016年报,主营业务分行业经营情况, 出现在页尾,且只有一行, 主营业务分行业情况,采用repair_list修复,不再采用这个
        mergedFieldsSecond = reduce(self._merge, secondFieldList)
        patternHeaderSecond = self._get_table_header_pattern(tableName,'headerSecond')
        isTableStartSecond = self._table_start_match(mergedFieldsSecond, patternHeaderSecond)

        isTableStartThird = False
        if thirdFieldList:
            mergedFieldsThird = reduce(self._merge, thirdFieldList)
            patternHeaderThird = self._get_table_header_pattern(tableName,'headerThird')
            isTableStartThird = self._table_start_match(mergedFieldsThird, patternHeaderThird)

        mergedHeaders = reduce(self._merge, headerList)
        patternHeaders = self._get_table_header_pattern(tableName,'headerFirst','headerSecond')
        isTableStartFourth = self._table_start_match(mergedHeaders,patternHeaders)

        tableStartScore = isTableStartFirst + isTableStartSecond + isTableStartThird + isTableStartFourth
        return int(tableStartScore)


    def _get_table_header_pattern(self,tableName,keyFirst,keySecond = NULLSTR):
        '''
        explain: 从表的配置中读取表头,转化为正则表达式
        args:
            keyFirst - 第一个表头名
            keySecond - 第二个表头名,默认为NULLSTR, 取其他值时,此时返回的pattern为两个表头拼接而成
        return:
            patternHeader - 表头的正则表达式
            1) keySecond为NULLSTR, patternHeader为keyFirst所指的表头转化为正则表达式
            2) keySecond不为NULLSTR, patternHeader为keyFirst和keySecond所值的表头拼接成正则表达式,拼接方法为内积
        '''
        assert keyFirst,'keyfirst must not be NULLSTR!'
        patternHeader = NULLSTR
        try:
            headerFirst = self.dictTables[tableName][keyFirst]
            # 在正则表达式中,'()[]'是元符号,需要替换成中文符号
            headerFirst = headerFirst.replace('(', '（').replace(')', '）').replace('[', '（').replace(']', '）')
            if headerFirst and keySecond:
                headerSecond = self.dictTables[tableName][keySecond]
                headerSecond = headerSecond.replace('(', '（').replace(')', '）').replace('[', '（').replace(']', '）')
                patternHeader = '|'.join(['^' + headerF + headerS for (headerF, headerS)
                                               in itertools.product(headerFirst.split('|'), headerSecond.split('|'))])
            elif headerFirst:
                # 针对keySecond 为NULLSTR的场景
                patternHeader = '|'.join(['^' + field for field in headerFirst.split('|')])
        except Exception as e:
            self.logger(e)
            raise ValueError(f'keyFirst:{keyFirst} or keySecond:{keySecond} is not a invalid key of table:{tableName}')
        return patternHeader


    def _table_start_match(self,mergedHeader,patternHeader):
        '''
        explain: 利用正则表达式匹配表头,匹配成功得一分
        args:
            mergedHeader - 表头字段的聚合
            patternHeader - 用于匹配表头字段的正则表达式
        return:
            isTableStartMatched - 表头匹配成功为True,反之为False
            1) 如果patternHeader和mergedHeader为NULLSTR,直接返回False
            2) 如果用patternHeader匹配到mergedHeader,返回True
        '''
        isTableStartMatched = False
        if isinstance(mergedHeader, str) and isinstance(patternHeader, str) \
            and patternHeader and mergedHeader:
            mergedHeader = mergedHeader.replace('(', '（').replace(')', '）').replace('[', '（').replace(']', '）')\
                .replace(' ', NULLSTR)
            matched = re.search(patternHeader, mergedHeader)
            if matched is not None:
                isTableStartMatched = True
        return isTableStartMatched


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
        if self.debugExtractTable == False:
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
        assert isinstance(prefix,str),"para(%s) of set_interpretPrefix must be string"%prefix
        self._interpretPrefix = prefix


    def initialize(self):
        self.loggingspace.clear_directory(self.loggingspace.directory)
        suffix = self.sourceFile.split('.')[-1]
        assert suffix.lower() in self.gConfig['pdfSuffix'.lower()], \
            'suffix of {} is invalid,it must one of {}'.format(self.sourceFile, self.gConfig['pdfSuffix'.lower()])


def create_object(gConfig):
    parser=DocParserPdf(gConfig)
    parser.initialize()
    return parser
