#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 12/9/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserSql.py
# @Note    : 用于sql数据库的读写
import re

from interpreterAccounting.docparser.docParserBaseClass import  *
import sqlite3 as sqlite
#from sqlalchemy import create_engine
import pandas as pd
from pandas import DataFrame
from ply import lex
import time

class Sqlite(SqilteBase):

    @Multiprocess.lock
    def _write_to_sqlite3(self, dataFrame: DataFrame, commonFields, tableName):
        conn = self._get_connect()
        dataFrame = dataFrame.T
        sql_df = dataFrame.set_index(dataFrame.columns[0], inplace=False).T
        isRecordExist = self._is_record_exist(conn, tableName, sql_df, commonFields)
        if isRecordExist:
            condition = self._get_condition(sql_df, commonFields)
            sql = ''
            sql = sql + 'delete from {}'.format(tableName)
            sql = sql + '\nwhere ' + condition
            self._sql_executer(sql)
            self.logger.info("delete from {} where is {} {} {}!".format(tableName, sql_df['公司简称'].values[0]
                                                                        , sql_df['报告时间'].values[0],
                                                                        sql_df['报告类型'].values[0]))
            sql_df.to_sql(name=tableName, con=conn, if_exists='append', index=False)
            conn.commit()
            self.logger.info("insert into {} where is {} {} {}!".format(tableName, sql_df['公司简称'].values[0]
                                                                        , sql_df['报告时间'].values[0],
                                                                        sql_df['报告类型'].values[0]))
        else:
            if sql_df['公司简称'].shape[0] > 0:
                sql_df.to_sql(name=tableName, con=conn, if_exists='append', index=False)
                conn.commit()
                self.logger.info("insert into {} where is {} {} {}!".format(tableName, sql_df['公司简称'].values[0]
                                                                            , sql_df['报告时间'].values[0],
                                                                            sql_df['报告类型'].values[0]))
            else:
                self.logger.error('sql_df is empty!')
        conn.close()

class LoggerParser(Logger):
    @classmethod
    def log_runinfo(cls, text='running '):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                result = func(self, *args, **kwargs)
                resultForLog = result
                columns = 0
                if isinstance(result, tuple):
                    resultForLog = result[0].T.copy()
                    columns = result[0].iloc[0]
                self.logger.info('%s %s() \n\t%s,%s%s,\t%s:\n\t%s\n\t%s\n\t columns=%s' %
                                 (text, func.__name__, self.dataTable['公司名称'], self.dataTable['报告时间']
                                  , self.dataTable['报告类型'], args[-1], '', resultForLog, columns))
                return result
            return wrapper
        return decorator


class DocParserSql(DocParserBase):
    def __init__(self,gConfig):
        super(DocParserSql, self).__init__(gConfig)
        self._create_tables()
        self.process_info = {}
        self.dataTable = {}
        self.dictLexers = self._construct_lexers()
        self.database = self.create_database(Sqlite)

    def _construct_lexers(self):
        dictLexer = dict()
        for tableName in self.dictTables.keys():
            dictLexer.update({tableName: dict()})
            for tokenName in ['field', 'header']:
                # 构建field lexer 和 header lexer
                dictTokens = self._get_dict_tokens(tableName, tokenName=tokenName)
                lexer = self._get_lexer(dictTokens)
                dictLexer[tableName].update({'lexer'+tokenName.title(): lexer, "dictToken" + tokenName.title(): dictTokens})
                self.logger.info('success to create %s lexer for %s!' % (tokenName, tableName))
        return dictLexer

    def _get_lexer(self, dictTokens):
        # Tokens
        # 采用动态变量名
        tokens = list(dictTokens.keys())
        local_name = locals()
        for token in tokens:
            local_name['t_' + token] = dictTokens[token]
        # self.logger.info(
        #    '%s:\n'%tableName + str({key: value for key, value in local_name.items() if key.split('_')[-1] in tokens}).replace("',","'\n"))

        t_ignore = " \n"

        def t_newline(t):
            r'\n+'
            t.lexer.lineno += t.value.count("\n")

        def t_error(t):
            print("Illegal character '%s'" % t.value[0])
            t.lexer.skip(1)

        # Build the lexer
        lexer = lex.lex(outputdir=self.workingspace.directory)
        return lexer

    def _get_dict_tokens(self, tableName, tokenName='field'):
        # 对所有的表字段进行标准化
        standardFields = self.dictTables[tableName][tokenName + 'Name']
        if len(standardFields) != len(set(standardFields)):
            self.logger.warning("the (fields/headers) of %s has duplicated:%s!" % (tableName, ' '.join(standardFields)))
        # 去掉标准化字段后的重复字段
        standardFields = list(set(standardFields))
        # 建立token和标准化字段之间的索引表
        fieldsIndex = dict([(tokenName.upper() + str(index), field) for index, field in enumerate(standardFields)])
        # 对字段别名表进行标准化,并去重
        virtualPassMatching = self.gJsonInterpreter['VIRTUALPASSMATCHING']
        dictAlias = self.dictTables[tableName][tokenName + 'Alias']
        if virtualPassMatching in dictAlias.values():
            fieldsIndex.update({'VIRTUALPASSMATCHING': virtualPassMatching})
        virtualStoping = self.gJsonInterpreter['VIRTUALSTOPING']
        if virtualStoping in dictAlias.values():
            fieldsIndex.update({'VIRTUALSTOPING': virtualStoping})
            standardFields = standardFields + [virtualStoping]
        # 判断fieldAlias中是否存在fieldName中不存在的字段,如果存在,则配置上存在错误.
        fieldDiff = set(dictAlias.values()).difference(set(standardFields))
        #对passMatching特殊处理,允许其在dictAlias中存在,但不包括在header中
        fieldDiff = fieldDiff.difference(set(list([virtualPassMatching,virtualStoping])))
        if len(fieldDiff) > 0:
            if NaN in fieldDiff:
                self.logger.warning('warning in (fieldAlias/headerAlias) of %s, NaN is exists' % tableName)
            else:
                self.logger.warning(
                    "warning in (fieldAlias/headerAlias) of %s,(field/header) not exists : %s"
                    % (tableName, ' '.join(list(fieldDiff))))
        # 在dictAlias中去掉fieldName中不存在的字段
        dictAlias = dict([(key, value) for key, value in dictAlias.items() if value not in fieldDiff])
        # 对fieldAlias和fieldName进行合并
        dictMerged = {}
        for key, value in dictAlias.items():
            dictMerged.setdefault(value, []).append(key)
        for key in standardFields:
            dictMerged.setdefault(key, []).append(key)

        # 最后把VIRTUALFIELD加上
        standardDiscardFields = self.dictTables[tableName][tokenName+'Discard']
        if len(standardDiscardFields) > len(set(standardDiscardFields)):
            # 如果fieldDiscard中在标准化后存在重复字段,则去重
            self.logger.warning(
                '%s has duplicated discardField after standardize:%s!' % (tableName, ' '.join(standardDiscardFields)))
            standardDiscardFields = list(set(standardDiscardFields))
        virtualField = self.gJsonInterpreter['VIRTUAL'+tokenName.upper()]
        fieldsIndex.update({'VIRTUAL'+tokenName.upper(): virtualField})
        # 增加一个默认值
        standardDiscardFields = standardDiscardFields + [virtualField]
        # 判断标准化后的fieldDiscard和Merged后的Key值是否还有重复,有则去掉
        fieldJoint = set(standardDiscardFields) & set(dictMerged.keys())
        if len(fieldJoint) > 0:
            if NaN in fieldJoint:
                self.logger.warning('%s (fieldDiscard/headerDiscard) has NaN field' % tableName)
            else:
                self.logger.debug(
                    "%s has dupicated (fieldDiscard/headerDiscard) with (fieldName/headerName):%s!"
                    % (tableName, ' '.join(list(fieldJoint))))
            standardDiscardFields = list(set(standardDiscardFields).difference(fieldJoint))
        for value in standardDiscardFields:
            if value is not NaN:
                dictMerged.setdefault(virtualField, []).append(value)

        # 构造dictTokens,token搜索的正则表达式即字段名的前面加上patternPrefix,后面加上patternSuffix
        patternPrefix = self.dictTables[tableName][tokenName + 'PatternPrefix']
        patternSuffix = self.dictTables[tableName][tokenName + 'PatternSuffix']
        dictTokens = dict()
        for token, field in fieldsIndex.items():
            # 生成pattern时要逆序排列,确保长的字符串在前面
            fieldList = sorted(dictMerged[field], key=lambda x: len(x), reverse=True)
            pattern = [patternPrefix + field + patternSuffix for field in fieldList]
            dictTokens.update({token: '|'.join(pattern)})
        return dictTokens


    def writeToStore(self, dictTable):
        self.dataTable = dictTable
        table = dictTable['table']
        tableName = dictTable['tableName']
        if len(table) == 0:
            self.logger.error('failed to process %s, the table row is empty'%tableName)
            return
        elif len(table[0]) <= 1:
            self.logger.error('failed to process %s, the table column is empty'%tableName)
            return

        self.process_info.update({tableName:{'processtime':time.time()}})
        self.process_info.update({'firstRowAllInvalid': dictTable['firstRowAllInvalid']})
        dataframe,countTotalFields = self._table_to_dataframe(table,tableName)#pd.DataFrame(table[1:],columns=table[0],index=None)

        #对数据进行预处理,两行并在一行的分开,去掉空格等
        dataframe = self._process_value_pretreat(dataframe,tableName)

        #针对合并所有者权益表的前三列空表头进行合并,对转置表进行预转置,使得其处理和其他表一致
        dataframe = self._process_header_merge_pretreat(dataframe, tableName)

        # 如果dataframe之有一行数据，说明从docParserPdf推送过来的数据是不正确的：只有表头，没有任何有效数据。
        if dataframe.shape[0] <= 1 or dataframe.shape[1] <= 1:
            self.logger.error("failed to process %s at start, the table has no data:%s,shape %s" % (
                tableName, dataframe.values, dataframe.shape))
            self.process_info.pop(tableName)
            return

        #把跨多个单元格的表字段名合并成一个
        dataframe = self._process_field_merge_simple(dataframe,'field',tableName)

        #去掉空字段及无用字段
        dataframe = self._process_field_discard(dataframe,'field' ,tableName)

        #同一张表的相同字段在不同财务报表中名字不同, 需要统一为相同名称, 统一后再去重
        dataframe = self._process_field_alias(dataframe,'field', tableName)

        #处理重复字段
        dataframe = self._process_field_duplicate(dataframe,'field',tableName)

        #对表进行转置,然后把跨多行的表头字段进行合并
        dataframe = self._process_header_merge_simple(dataframe,tableName)

        #去掉无用的表头;同时对水平表进行转置,把字段名由index转为column
        dataframe = self._process_header_discard(dataframe, tableName)

        # 同一张表的相同表头在不同财务报表中名字不同, 需要统一为相同名称, 统一后再去重
        dataframe = self._process_header_alias(dataframe, tableName)
        #把表头进行标准化

        # 把表头的重复字段进行区分, 去掉非法的字段,需要处理 header为 xxxx年的情况才能启用
        # 功能测试通过,但暂不做处理,便于从入库后数据中更易发现问题
        #dataframe = self._process_header_duplicate(dataframe,tableName)

        #去掉不必要的行数据,比如合并资产负债表有三行数据,最后一行往往是上期数据的修正,是不必要的
        dataframe = self._process_row_tailor(dataframe, tableName)

        #如果dataframe之有一行数据，说明从docParserPdf推送过来的数据是不正确的：只有表头，没有任何有效数据。
        if dataframe.shape[0] <= 1 or dataframe.shape[1] <= 1:
            self.logger.error("failed to process %s, the table has no data:%s,shape %s"%(tableName,dataframe.values,dataframe.shape))
            self.process_info.pop(tableName)
            return

        #dataframe前面插入公共字段
        dataframe = self._process_field_common(dataframe, dictTable, countTotalFields,tableName)

        dataframe = self._process_value_standardize(dataframe,tableName)

        #把dataframe写入sqlite3数据库
        reportType = self.standard._get_report_type_by_filename(self.gConfig['sourcefile'])
        targetTableName = self.standard._get_tablename_by_report_type(reportType, tableName)
        #with self.processLock:
            # 多进程写数据库时必须加锁
        self.database._write_to_sqlite3(dataframe,self.commonFields, targetTableName)
        self.process_info[tableName].update({'processtime':time.time() - self.process_info[tableName]['processtime']})


    #@loginfo()
    def _table_to_dataframe(self,table,tableName):
        horizontalTable = self.dictTables[tableName]['horizontalTable']
        if horizontalTable == True:
            #对于装置表,如普通股现金分红情况表,不需要表头
            dataFrame = pd.DataFrame(table,index=None)
            countTotalFields = len(dataFrame.columns.values)
        else:
            dataFrame = pd.DataFrame(table, index=None)
            countTotalFields = len(dataFrame.index.values)
        dataFrame.fillna(NONESTR,inplace=True)
        return dataFrame,countTotalFields


    def _rowPretreat(self,row):
        self.lastValue = None
        row = row.apply(self._valuePretreat)
        return row


    def _valuePretreat(self,value):
        """
        args:
            value - dataFrame中的某一个字段数据
        return:
            value - 经过预处理后的字段数据,处理原则:
            1) 去掉所有的中文, 避免_is_header_in_row误判.
            2) 奥美医疗2018年年报,主要会计数据中,存在两列数值并列到了一列,同时后接一个None的场景, 把一个字段拆成两个字段.
        """
        try:
            if isinstance(value,str):
                if value != NONESTR and value != NULLSTR:
                    value = re.sub('不适用$',NULLSTR,value)
                    value = re.sub('^附\\s*注[一二三四五六七八九十〇]*',NULLSTR,value)#解决隆基股份2017年报中,合并资产负债表中的出现"附注六、1"； 解决海通证券：2020年半年度报告 合并资产负债表 出现 附注五
                    value = re.sub('^[一二三四五六七八九十〇、]+附\\s*注', NULLSTR,value)  # 解决厦门钨业 2016年报,合并利润表表中的出现"七、附注61"
                    value = re.sub('元$',NULLSTR,value)#解决海螺水泥2018年报中,普通股现金分红情况表中出现中文字符,导致_process_field_merge出错
                    value = re.sub('^上升了', NULLSTR, value)  # 解决圣农发展2016年报,主营业务分行业经营情况出现 "上升 了0.49个百分点"
                    value = re.sub('^上升',NULLSTR,value)#解决京新药业2017年年报,主要会计数据的一列数据中出现"上升 0.49个百分点"
                    value = re.sub('个百分点$',NULLSTR,value)#解决京新药业2017年年报,海螺水泥2018年年报主要会计数据的一列数据中出现"上升 0.49个百分点"
                    value = re.sub('个.*百分点$',NULLSTR,value)#解决 许继电气 2019年报,主营业务分行业经营情况中出现 : 个% 百分点
                    value = re.sub('个百分$', NULLSTR, value)  # 解决新城控股2015年年报,主营业务分行业经营情况一行数据在末尾"减少 4.38 个百分"
                    value = re.sub('^注释',NULLSTR,value)#解决灰顶科技2019年年报中,合并资产负债表,合并利润表中的字段出现'注释'
                    value = re.sub('^）\\s*',NULLSTR,value)
                    value = re.sub('^同比增加',NULLSTR,value)#解决冀东水泥：2017年年度报告.PDF主要会计数据中的一列数据中出现'同比增加',导致_precess_header_merge_simple误判
                    value = re.sub('注$', NULLSTR, value)#解决康泰生物2018年年报中普通股现金分红情况表中出现中文字符'注',导致_process_merge_header_simple出问题
                    value = re.sub('^增\\s*加', NULLSTR, value) #1)解决海天味业2014年报中主营业物质的数据中出现,'增加','减少'； 2)解决福耀玻璃2015年报 主营业务分行业经营情况的数据中出现: 增 加 0.27 个百分点
                    value = re.sub('^减\\s*少', '-', value) #解决海天味业2014年报中主营业物质的数据中出现,'增加','减少'
                    value = re.sub('^下降[了]*', '-', value)  # 解决海螺水泥2014年报中主营业物质的数据中出现,'下降'; 解决 圣农发展：2020年年度报告,主营业务分行业经营情况出现''下降了
                    value = re.sub('^储$', NULLSTR, value) #解决尚荣医疗2014年报,合并所有者权益变动表,、上年年末余额 这一行中出现 "储","备
                    value = re.sub('^备-$',NULLSTR, value)
                    value = re.sub('^无$', NULLSTR, value) #解决海康威视2015年报,普通股现金分红情况表 最后一列出现 '无'
                    value = re.sub('^增长', NULLSTR, value) #解决荣盛发展2018年报,主营业务分行业经营情况,最后一列数据中出现'增长'
                    value = re.sub('【注.*】',NULLSTR,value) #解决健友股份：2018年 主营业务分行业经营情况 的数据中出现 【注 1】
                    value = re.sub('注\\d+$',NULLSTR,value) #解决巨人网络：2019年 主要会计数据 的数据中出现 注 1
                    value = re.sub('^不派发现金红利$',NULLSTR,value) #解决沪电股份2014年报 普通股现金分行情况表中出现 '不派发现金红利'
                    value = re.sub('^每\\d+股转增\\d+股',NULLSTR,value) #解决（300033）同花顺：2014年年度报告中,普通股现金分行情况表中出现 每10股转增10股
                    value = re.sub('^货币单位：人民币$',NULLSTR,value) # 解决思源电器2015年报,合并资产负债表的前三行为 : 编制单位：思源电气股份有限公司合并资产负债表2015年12月31日 货币单位：人民
                    value = re.sub('^去年同期为负$',NULLSTR,value) # 解决英科医疗2020年年报,主要会计数据, 归属于上市公司股东的扣除非经常性损益的净利润（元）这一行中出现: 去年同期为负
                    value = re.sub('(?=\\d)\\d、$',NULLSTR,value) # 解决双汇发展2020年报,主要会计数据,主营业收入一行中出现: 1、
                    #解决海康威视2019-2014年报合并资产负债表等 中的辅助中出现 :  (五)2
                    value = re.sub('(^\\s*[（(]*[一二三四五六七八九十〇、]{1,4})[)）]*(?=[^\\u4E00-\\u9FA5])', NULLSTR, value) # 新城控股合并利润表中出现附注,数据中出现四(38)及一、四(38).  海康威视2019-2014年报合并资产负债表等 中的附注中出现 :  (五)2
                    value = re.sub('([（(]*[一二三四五六七八九十〇、]{1,4})[)）]*(?=[^\\u4E00-\\u9FA5])', NULLSTR, value)  # 执行两次,解决安琪酵母2014年报中,并资产负债表等 中的附注中出现 :五（一）.
                    value = re.sub('[一二三四五六七八九十〇、]{1,4}', NULLSTR, value)# 解决上海贝岭2018年合并利润表,出现 七、(五二)
                    #解决高德红外2014年报中 主营业务分行业经营情况表中,出现（%）,改为通过headerStandardize来解决
                    # 解决五粮液2020年年报,主要会计数据中,存在两列数据并列到了一列,同时后接一个None的场景和,其中总资产字段合并为一列时中间间隔1个空字符,暂通过repair_list解决 at 20210612
                    result = re.split("[ ]{2,}(?=[\\d-])",value,maxsplit=1)
                    if len(result) > 1:
                        value,self.lastValue = result
                    value = value.replace(' ',NULLSTR)       # 解决石头科技2019年报,主营业务分行业经营情况中出现 "增加 7.30个 百 分 点" 和 "增加 8.65个 百 分"
                    value = re.sub('个百分点$', NULLSTR, value)
                    value = re.sub('个百分$', NULLSTR, value)
                    value = re.sub('个$', NULLSTR, value)  # 深圳燃气：2018年年度报告,主营业务分行业经营情况, 被拆成2页, 第一页判断为结尾,出现: 减少 2.35 个
                else:
                    if self.lastValue != None and value == NONESTR:
                        # 解决奥美医疗2018年年报,主要会计数据中,存在两列数值并列到了一列,同时后接一个None的场景
                        value = self.lastValue
                    self.lastValue = None
        except Exception as e:
            print(e)
        return value


    def _process_value_pretreat(self,dataFrame:DataFrame,tableName):
        """
        args:
            dataFrame - 待处理的表数据
            tableName - 表名, 用于从dictTable中作为索引读取配置参数
        return:
            dataFrame - 经过预处理后的表数据,处理原则:
            1) 从第一行开始搜索, 跳过字段全空的行 及 包含表头的行, 从第一行数据开始, 调用_rowPretreat,对数据进行预处理,主要是去除数据中所有的文字,
               如:附注,百分比等. 针对海通证券2020年半年度报告,合并资产负债表,第二行有个附注, 这个附注必须留着, 采用_process_header_discard整列去除.
            2) 对所有的数据用_replace_value进行处理, 主要对 () 替换为中文的 （）,去掉全部空格
        """
        #采用正则表达式替换空字符,对一个字段中包含两个数字字符串的进行拆分
        #解决奥美医疗2018年年报,主要会计数据中,存在两列数值并列到了一列,同时后接一个None的场景.
        #东材科技2018年年报,普通股现金分红流量表,表头有很多空格,影响_process_header_discard,需要去掉
        # 海通证券：2020年半年度报告.PDF 合并资产负债表, 表头分布在第二行, 第二行有两个字段是 附注, 需要保留
        index = 0
        currentRow = dataFrame.iloc[index]
        self.lastValue = None
        while (dataFrame.shape[0] > 0 and index + 1 < dataFrame.shape[0]) \
            and (self._is_row_all_invalid(currentRow.tolist()) or self._is_header_in_row(currentRow.tolist(),tableName)):
            self.logger.info('skip all invalid row or no header row: %s' % currentRow.tolist())
            index = index + 1
            currentRow = dataFrame.iloc[index]
            currentRow = currentRow.apply(self._valuePretreat)
        dataFrame.iloc[index:,1:] = dataFrame.iloc[index:,1:].apply(self._rowPretreat, axis=1)
        #dataFrame.iloc[1:,1:]  = dataFrame.iloc[1:,1:].apply(self._rowPretreat,axis=1)
        dataFrame.iloc[:,:] = dataFrame.iloc[:,:].apply(lambda row:row.apply(self._replace_value))
        return dataFrame


    def _process_header_merge_pretreat(self, dataFrame, tableName):
        isHorizontalTable = self.dictTables[tableName]['horizontalTable']
        fieldFromHeader = self.dictTables[tableName]['fieldFromHeader']
        mergedRow = None
        lastIndex = 0
        # 增加blankFrame来驱动最后一个field的合并
        #blankFrame = pd.DataFrame([''] * len(dataFrame.columns.values), index=dataFrame.columns).T
        #dataFrame = dataFrame.append(blankFrame)
        while dataFrame.shape[0] > 0 and ( self._is_row_all_invalid(dataFrame.iloc[0].tolist()) \
            or self._is_header_in_row(dataFrame.iloc[0].tolist(),tableName) == False):
            # self._is_header_in_row(dataFrame.iloc[0].tolist(),tableName) == False 解决思源电器2015年报,合并资产负债表,前三行为: 编制单位：思源电气股份有限公司合并资产负债表2015年12月31日货币单位：人民
            # 如果第一行数据全部为无效的,则删除掉. 解决康泰生物：2016年年度报告.PDF,合并所有者权益变动表中第一行为全None行,导致标题头不对的情况
            # 但是解析出的合并所有者权益变动表仍然是不对的,原因是合并所有者权益变动表第二页的数据被拆成了两张无效表,而用母公司合并所有者权益变动表的数据填充了.
            self.logger.info('delete all invalid row or no header row: %s' % dataFrame.iloc[0].tolist())
            dataFrame.iloc[0] = NaN
            dataFrame = dataFrame.dropna(axis=0)
        dataFrame = self._major_accounting_data_tailor(dataFrame,tableName)
        if dataFrame.shape[0] == 0:
            return dataFrame
        for index, field in enumerate(dataFrame.iloc[:, 0]):
            isRowNotAnyNone = self._is_row_not_any_none(dataFrame.iloc[index],tableName)
            isHeaderInRow = self._is_header_in_row(dataFrame.iloc[index].tolist(),tableName)
            isHeaderInMergedRow = self._is_header_in_row(mergedRow,tableName)
            isRowAllInvalid = self._is_row_all_invalid_exclude_blank(dataFrame.iloc[index])
            if isRowAllInvalid == False:
                if isHeaderInRow == False:
                    #表字段所在的行,清空合并行
                    if isHeaderInMergedRow:
                        if index > lastIndex + 1:
                            dataFrame.iloc[lastIndex] = mergedRow
                            dataFrame.iloc[lastIndex + 1:index] = NaN
                        mergedRow = None
                    elif isHorizontalTable == True and len(fieldFromHeader) > 0:
                        # 康泰生物2019年年报普通股现金分红情况表,有一列全部为None,此时isRowNotAnyNone失效
                        # 针对主营业务分行业经营情况做特殊处理,因为isHorizontalTable=True,同时fieldFromHeader非空情况下,只有这张表
                        if isRowNotAnyNone == True:
                            if index > lastIndex + 1:
                                dataFrame.iloc[lastIndex] = mergedRow
                                dataFrame.iloc[lastIndex + 1:index] = NaN
                            mergedRow = None
                    else:
                        mergedRow = None
                else:
                    if isHeaderInMergedRow == False:
                        # 解决再升科技2018年年报,合并所有者权益变动表在每个分页中插入了表头
                        # 解决大立科技：2018年年度报告,有一行", , , ,调整前,调整后, , , , ",满足isRowNotAnyNone==True条件,但是需要继续合并
                        mergedRow = None

            if mergedRow is None:
                mergedRow = dataFrame.iloc[index].tolist()
                lastIndex = index
            else:
                mergedRow = self._get_merged_row(dataFrame.iloc[index].tolist(), mergedRow, isFieldJoin=True)

        if isHorizontalTable == True:
            # 解决（300326）凯利泰：2018年年度报告.PDF，普通股现金分红请表，除了第一行解析正常为“”
            if len(fieldFromHeader) == 0:
                #针对恒瑞医药2014年报,普通股现金分红情况表,起分红年度为:2014,2013,2012,需要补充为2014年,2013年,2012年
                dataFrame.iloc[:,0] = dataFrame.iloc[:,0].apply(self._fill_year_standardize)
            dataFrame = dataFrame.dropna(axis=0)
            #把第一列做成索引
            dataFrame.set_index(0,inplace=True)
            dataFrame = dataFrame.T.copy()
        else:
            # 国金证券2016年报,丽珠集团：2014年年度报告, 合并资产负债表的表头出现 : 2015.12.31  2014.12.31
            dataFrame.iloc[0] = dataFrame.iloc[0].apply(self._fill_year_standardize)
            dataFrame.columns = dataFrame.iloc[0].copy()
            #主要会计数据的表头通过上述过程去不掉,采用专门的函数去掉
            dataFrame = self._discard_header_row(dataFrame,tableName)
        return dataFrame


    @LoggerParser.log_runinfo()
    def _process_field_merge_simple(self,dataFrame:DataFrame,tokenName,tableName):
        # 解决康泰生物2019年年报,主要会计数据解析不正确,每行中都出现了None
        # 解决贝达药业2018年年报无形资产情况表解析不正确
        # 解决大立科技2018年财报中主要会计数据解析不准确的问题,原因是总资产(元)前面接了一个空字段,空字段的行需要合并到下一行中
        # 解决康泰生物2019年年报,主要会计数据解析不正确,每行中都出现了None
        #增加一行空白行,以推动最后一个字段的合并
        blankFrame = pd.DataFrame(['']*len(dataFrame.columns.values),index=dataFrame.columns).T
        dataFrame = dataFrame.append(blankFrame)
        # 解决合并利润表中某些字段的'-'符合使用不一致,统一替换城'－'
        #dataFrame.iloc[:,0] = self._fieldname_pretreat(dataFrame.iloc[:,0])
        # 将所有正则表达式中要用到的字符全部替换成中文字符
        dataFrame.iloc[:,0] = dataFrame.iloc[:,0].apply(self._replace_fieldname)
        #row = row.apply(lambda value: value.replace('-', '－').replace('(', '（').replace(')', '）').replace('.', '．'))
        mergedColumn = reduce(self._merge, dataFrame.iloc[:, 0].tolist())
        countTotal = len(mergedColumn)
        lexer = self.dictLexers[tableName]['lexer'+tokenName.title()]
        dictToken = self.dictLexers[tableName]['dictToken'+tokenName.title()]
        lexer.input(mergedColumn)
        dictFieldPos = dict()
        for index,tok in enumerate(lexer):
            #针对主营业务分行业经营情况表做特殊处理
            dictFieldPos.update({index:{'lexpos':tok.lexpos,'value':tok.value,'type':tok.type}})
            self.logger.info('%s the %s lexer matched the field:%d %s %s %s\t%s'%(tableName,tokenName,index,tok.lexpos,tok.value,tok.type,dictToken[tok.type]))
            if self.dictTables[tableName]['horizontalTable'] == True:
                if tok.type == 'VIRTUALSTOPING':
                    #对于水平表,主要是主营业务分行业经营情况, 目前只解析到 分行业的情况,对于分产品及之后的字段不再合并,设置VIRTUALSTOPING = '分产品'
                    # 加1的目的是让后续的字段合并永远达不到最后一个字段,即VIRTUALSTOPING之后的字段就不再做合并
                    countTotal = countTotal + 1
                    break

        countPos = len(dictFieldPos.keys())
        if countPos <= 0:
            self.logger.error('%s failed to use lexer %s !'%(tableName,mergedColumn))
            return dataFrame
        posIndex = 0
        fieldPos = 0
        mergedRow = None
        lastIndex = 0
        lexPos = dictFieldPos[posIndex]['lexpos']
        for index,field in enumerate(dataFrame.iloc[:,0].tolist()):
            isRowNotAnyNone = self._is_row_not_any_none(dataFrame.iloc[index],tableName)
            while fieldPos > lexPos and lexPos < countTotal:
                # 字段位置超前了,lexPos追平
                posIndex = posIndex + 1
                if posIndex < countPos:
                    lexPos = dictFieldPos[posIndex]['lexpos']
                    #追上来的过程中,以前合并的字段就不要了
                    mergedRow = None
                else:
                    lexPos = countTotal
                    break

            if fieldPos == lexPos:
                if fieldPos > lexPos and posIndex < countPos:
                    #如果fieldPos 比 lexPos大,说明lex没有匹配到字段开头,则存在错误
                    self.logger.error("failed to match the whole (field/header) %s %s match %s"%(tableName,dictFieldPos[posIndex]['value'],field))
                #说明在该位置搜索到了字段
                if self._is_valid(field):
                    if index > lastIndex + 1 and mergedRow is not None:
                        # 把前期合并的行赋值到dataframe的上一行
                        dataFrame.iloc[lastIndex] = mergedRow
                        dataFrame.iloc[lastIndex + 1:index] = NaN
                    if mergedRow is not None and mergedRow[0] == NULLSTR:
                        # 解决华东医疗2015年年报,主营业务分行业经营情况表,主营业务收入在第二行,前面一行的第一个字段为空字段'',需要和第二行进行合并
                        pass
                    else:
                        mergedRow = None
                    #开始搜寻下一个字段
                    posIndex = posIndex + 1
                else:
                    # 正对字段名是None和NULLSTR的处理,这两种情况下,该字段不占用字符串长度
                    # field == 'None' 解决 天山股份：2018年半年度报告,合并所有者权益变动表, 数据起始行第一个字段为None: None 1,048,722 4,054,364,3	180,102,29	39,200,830.	271,961,81	1,756,703,4	915,452,04	8,266,507,7
                    if (field == NULLSTR or field == 'None') and isRowNotAnyNone:
                        #如果是空字段,而且没有一个是None,说明是全空白字段,或者带有数值,该段要向下一个字段合并,并且把前面的mergedRow清空
                        if index > lastIndex + 1 and mergedRow is not None:
                            # 把前期合并的行赋值到dataframe的上一行
                            dataFrame.iloc[lastIndex] = mergedRow
                            dataFrame.iloc[lastIndex + 1:index] = NaN
                        mergedRow = None
                        # 开始搜寻下一个字段
                        posIndex = posIndex + 1

            #更新字段位置和lex位置
            if posIndex < countPos:
                lexPos = dictFieldPos[posIndex]['lexpos']
            else:
                lexPos = countTotal
            fieldLen = self._get_field_len(field)
            fieldPos = fieldPos + fieldLen

            if mergedRow is None:
                mergedRow = dataFrame.iloc[index].tolist()
                lastIndex = index
            else:
                mergedRow = self._get_merged_row(dataFrame.iloc[index].tolist(), mergedRow, isFieldJoin=True)
        return dataFrame


    def _process_header_merge_simple(self,dataFrame:DataFrame, tableName):
        #必须采用这种方式set_index,如果采用set_index(dataFrame.columns[0])方式,当columns[0]=NULLSTR时,转换出来的结果是错误的.
        dataFrame.set_index(dataFrame.iloc[:,0],inplace=True)
        dataFrame = dataFrame.drop(dataFrame.columns[0],axis=1)
        # 转置的时候第一个header会丢失,必须通过coluns.name方式找回来
        dataFrame.columns.name = dataFrame.index.name
        if tableName == '主营业务分行业经营情况':
            self.logger.info("%s 分行业,分产品,分区域列表如下,just for debug:\n%s"
                             %(tableName,"\n".join([dataFrame.columns.name] + dataFrame.columns.tolist())))
        dataFrame = dataFrame.T.reset_index()

        isHorizontalTable = self.dictTables[tableName]['horizontalTable']
        if isHorizontalTable == False:
            #对于非转置表,在_process_header_merge_pretreat中实际对header字段做了合并,除非对字段做标准化,否则暂时不需要再次进行合并
            #return dataFrame
            pass
        dataFrame = self._process_field_merge_simple(dataFrame,"header",tableName)
        return dataFrame


    @LoggerParser.log_runinfo()
    def _process_field_common(self, dataFrame, dictTable, countFieldDiscard,tableName):
        #在dataFrame前面插入公共字段
        fieldFromHeader = self.dictTables[tableName]["fieldFromHeader"]
        countColumns = len(dataFrame.columns) + countFieldDiscard
        index = 0
        for (commonFiled, _) in self.commonFields.items():
            if commonFiled == "ID":
                #跳过ID字段,该字段为数据库自增字段
                continue
            if commonFiled == "报告时间":
                assert dictTable[commonFiled] != NULLSTR,'dictTable[%s] must not be null!'%commonFiled
                #公共字段为报告时间时,需要特殊处理
                if len(fieldFromHeader) != 0:
                    #针对分季度财务指标,指标都是同一年的,但是分了四个季度
                    value =  [commonFiled,*[str(int(dictTable[commonFiled].split('年')[0])) + '年'
                                       for i in range(len(dataFrame.iloc[:,0])-1)]]
                else:
                    firstHeader = dataFrame.index.values[0]
                    #针对普通股现金分红情况
                    if isinstance(firstHeader,str) and (firstHeader == '分红年度' or firstHeader == '年度'):
                        #firstHeader == '年度'是为了解决海螺水泥2018年年报普通股现金分红情况表中,表头是年度
                        value = [commonFiled,*dataFrame.index[1:].tolist()]
                    elif isinstance(firstHeader, str) and firstHeader == '主要会计数据' \
                        and self.standard._is_year_list(dataFrame.index[1:]):
                        # firstHeader == '主要会计数据' 是为了解决国检集团2019年年报主要会计数据中, 只有2019年,2017年数据被留下来了
                        value = [commonFiled, *dataFrame.index[1:].tolist()]
                    else:
                        value = [commonFiled,*[str(int(dictTable[commonFiled].split('年')[0]) - i) + '年'
                                       for i in range(len(dataFrame.iloc[:,0])-1)]]
            else:
                value = [commonFiled,*[dictTable[commonFiled]]*(len(dataFrame.iloc[:,0])-1)]
            dataFrame.insert(index,column=countColumns,value=value)
            countColumns += 1
            index += 1
        dataFrame = self._process_field_from_header(dataFrame,fieldFromHeader,index,countColumns,tableName)
        return dataFrame


    def _process_value_standardize(self,dataFrame,tableName):
        #对非法值进行统一处理
        def valueStandardize(value):
            try:
                if isinstance(value,str):
                    value = value.replace('\n', NULLSTR).replace(' ', NULLSTR).replace(NONESTR,NULLSTR)\
                        .replace('/',NULLSTR)
                    #海天味业2015年报:1)现金流量表补充资料, (减少以“ ( ) ”号填列) 2)合并所有者权益变动表,(减少以“ ( ) ”号填列)
                    value = re.sub(r'（([\d.,]+)）','-\g<1>',value)
                    value = value.replace('）',NULLSTR).replace('（',NULLSTR)
                    #解决迪安诊断2018年财报主要会计数据中,把最后一行拆为"归属于上市公司股东的净资产（元"和"）"
                    #高德红外2018年报,无效值用'--'填充,部分年报无效值用'-'填充,万科A 2020年半年报,合并所有者权益变动表中的无效值用 '---'填充
                    value = re.sub('.*-$',NULLSTR,value)
                    #尚荣医疗2017年年报,现金流量表补充资料中的无效值用 '一'填充
                    value = re.sub('^一$',NULLSTR,value)
                    # 尚荣医疗2018年年报,现金流量表补充资料中的无效值用 '一'填充
                    value = re.sub('^—$', NULLSTR, value)
            except Exception as e:
                print(e)
            return value
        dataFrame.iloc[1:] = dataFrame.iloc[1:].apply(lambda x: x.apply(valueStandardize))
        return dataFrame


    def _process_field_from_header(self,dataFrame,fieldFromHeader,index,countColumns,tableName):
        assert isinstance(fieldFromHeader,list),"parameter fieldFromHeader must be a list!"
        isHorizontalTable = self.dictTables[tableName]['horizontalTable']
        #在公共字段后插入由表头转换来的字段
        if len(fieldFromHeader) != 0:
            values = dataFrame.index.values.tolist()
            if isHorizontalTable == True:
                #isHorizontalTable=True,fieldFromHeader有效的情况下,只有主营业务分行业经营情况表,需要做特殊处理
                insertValues = list([NULLSTR]*len(values))
                dictFieldPoses = dict([(values.index(field),field) for field in fieldFromHeader if field in values])
                fieldPoses = sorted(list(dictFieldPoses.keys()))
                if len(fieldPoses) >= 1:
                    for posIndex,fieldPos in enumerate(fieldPoses[1:]):
                        insertValues[fieldPoses[posIndex]:fieldPos] = values[fieldPoses[posIndex]:fieldPos]
                        insertValues[0] = dictFieldPoses[fieldPoses[posIndex]]
                        if fieldPoses[posIndex] != 0:
                            insertValues[fieldPoses[posIndex]] = NaN
                        dataFrame.insert(index, column=countColumns, value=insertValues)
                        insertValues = list([NULLSTR] * len(values))
                        index = index + 1
                        countColumns = countColumns + 1
                    #最后一个字段的处理
                    insertValues[fieldPoses[-1]:] = values[fieldPoses[-1]:]
                    insertValues[0] = dictFieldPoses[fieldPoses[-1]]
                    if fieldPoses[-1] != 0:
                        insertValues[fieldPoses[-1]] = NaN
                    dataFrame.insert(index, column=countColumns, value=insertValues)
                    dataFrame = dataFrame.dropna(axis=0).copy()
                    if len(fieldPoses) == len(fieldFromHeader):
                        self.logger.info('%s: success to process all field from header,%s all in %s' % (tableName, fieldFromHeader, values))
                    else:
                        self.logger.warning('%s: success to process %d field from header,but %s not all in %s' % (tableName,len(fieldPoses), fieldFromHeader, values))
                else:
                    self.logger.error('%s: failed to process field from header,%s not in %s'%(tableName,fieldFromHeader,values))
            else:
                values[0] = fieldFromHeader[0]
                dataFrame.insert(index,column=countColumns,value=values)
        return dataFrame


    def _process_field_duplicate(self,dataFrame,tokenName,tableName):
        # 重复字段处理,放在字段标准化之后
        duplicatedFields = self._get_duplicated_field(dataFrame.iloc[:,0].tolist())
        standardizedFields = self.dictTables[tableName][tokenName + 'Name']
        duplicatedFieldsStandard = self._get_duplicated_field(standardizedFields)
        dataFrame.iloc[:,0] = duplicatedFields
        duplicatedFieldsResult = []
        for field in duplicatedFields:
            if field in duplicatedFieldsStandard:
                duplicatedFieldsResult += [field]
            else:
                self.logger.warning('failed to add field: %s is not exist in %s'%(field,tableName))
                #删除该字段
                indexDiscardField = dataFrame.iloc[:,0].isin([field])
                discardColumns = indexDiscardField[indexDiscardField == True].index.tolist()
                dataFrame.loc[discardColumns] = NaN
                dataFrame = dataFrame.dropna(axis=0).copy()
        return dataFrame


    def _process_header_duplicate(self, dataFrame, tableName):
        fieldFromHeader = self.dictTables[tableName]['fieldFromHeader']
        if len(fieldFromHeader) > 0 and tableName != '主营业务分行业经营情况':
            # 仅仅对header要作为数据库入库字段的才进行处理,fieldFromHeader表示要把表头转成field 并写入数据库,这种情况需要去重处理,同时去掉非法字段
            # 主营业务分行业经营情况 不纳入考虑,原因是只对 分行业 的header进行了标准化,分产品,分地区没有做标准化,但是也还是要写道数据库中
            dataFrame = self._process_field_duplicate(dataFrame,'header',tableName)
        return dataFrame


    def _process_header_discard(self, dataFrame, tableName):
        dataFrame = self._process_field_discard(dataFrame,'header',tableName)
        return dataFrame


    def _process_field_discard(self, dataFrame, tokenName,tableName):
        #去掉空字段,针对主要会计数据这张表,需要提出掉其空字段
        #对于普通股现金分红情况表,则忽略这一过程
        fieldDiscard = self.dictTables[tableName][tokenName + 'Discard']
        fieldDiscardPattern = '|'.join(fieldDiscard)
        indexDiscardField = [self._is_field_matched(fieldDiscardPattern, x) for x in dataFrame.iloc[:,0]]
        # 主要会计数据的第一个index为空字段,该代码会判断为Ture,所以要重新设置为False
        dataFrame.loc[indexDiscardField] = NaN
        #对非中文字符全部替换为中文字符,这部分工作已经在_fields_replace_punctuate完成
        indexDiscardField = dataFrame.iloc[:,0].isin(self._get_invalid_field())
        dataFrame.loc[indexDiscardField] = NaN
        dataFrame = dataFrame.dropna(axis=0).copy()
        return dataFrame


    def _process_header_alias(self,dataFrame,tableName):
        #把表头进行统一化,标准化
        dataFrame = self._process_field_alias(dataFrame,'header',tableName)
        return dataFrame


    def _process_field_alias(self, dataFrame,tokenName,tableName):
        #同一张表的相同字段在不同财务报表中名字不同,需要统一为相同名称
        #针对主要会计数据,需要在标准化前进行统一命名
        #对于合并利润表,需要在标准化后进行统一命名
        standardizedFields = self._get_standardized_keyword(dataFrame.iloc[:,0].tolist(),self.dictTables[tableName][tokenName+'Standardize'])
        dataFrame.iloc[:,0] = standardizedFields
        aliasedFields = self._get_aliased_fields(dataFrame.iloc[:,0].tolist(),tokenName, tableName)
        dataFrame.iloc[:,0] = aliasedFields
        if tokenName == 'header':
            # 对columns[0]做特殊处理
            column0 = dataFrame.columns.values[0]
            standardizedColumns0 = self._get_standardized_keyword(column0,self.dictTables[tableName][tokenName + 'Standardize'])
            aliasedColumns0 = self._get_aliased_fields(standardizedColumns0, tokenName, tableName)
            if aliasedColumns0 is not NaN:
                dataFrame.columns.values[0] = aliasedColumns0
        #统一命名之后,再次进行标准化
        dataFrame = dataFrame.dropna(axis = 0).copy()
        return dataFrame


    def _process_row_tailor(self, dataFrame, tableName):
        #转化成需要的dataFrame
        dataFrame = dataFrame.T.reset_index().T
        dataFrame.set_index(dataFrame.columns[0],inplace=True)
        #maxHeaders = self.dictTables[tableName]['maxHeaders']
        maxHeaders = self._get_max_headers(tableName)
        fieldFromHeader = self.dictTables[tableName]['fieldFromHeader']
        if len(fieldFromHeader) == 0:
            #对于合并所有者权益变动表,无形资产情况,分季度主要财务数据等表,不需要对多余的行进行裁剪
            maxRows = len(dataFrame.index.values)
            if maxRows > maxHeaders:
                if self._is_third_quarter_accounting_data(tableName):
                    #对于第三季度报告的主要会计数据,只保留最后一行,去掉第一行数据
                    self.logger.warning(
                        "%s has %d row,only %s is needed,row %s had discarded!" % (tableName, maxRows, maxHeaders
                                                                                   ,dataFrame.index[-2]))
                    if '上期金额' in dataFrame.index.values:
                        dataFrame.loc['上期金额'] = NaN
                    else:
                        dataFrame.iloc[ - 2] = NaN
                    dataFrame = dataFrame.dropna(axis = 0).copy()
                elif self._is_third_quarter_profit_data(tableName):
                    if '本报告期金额' in dataFrame.index.values:
                        # 解决恒顺醋业,第三季度报告,合并利润表,中有四列数据, 取倒数第二列数据,为年初至报告期的数据
                        dataFrame.loc['本期金额'] = dataFrame.loc['本报告期金额']
                    elif len(dataFrame.index.values) == 5:
                        dataFrame.loc['本期金额'] = dataFrame.iloc[-2]
                    else:
                        self.logger.error('dataFrame has no columns 本报告期金额, whitch is %s'% dataFrame.index.values)
                    self.logger.warning(
                        "%s has %d row,only %s is needed,last row %s had discarded!"%(tableName,maxRows,maxHeaders
                                                                                      ,dataFrame.index[-1]))
                    dataFrame.iloc[maxHeaders:maxRows] = NaN
                    dataFrame = dataFrame.dropna(axis=0).copy()
                else:
                    #对于超出最大头长度的行进行拆解,解决华侨城A 2018年包中,合并资产负债表 有三列数据,其中最后一列数据是不需要的
                    self.logger.warning(
                        "%s has %d row,only %s is needed,last row %s had discarded!"%(tableName,maxRows,maxHeaders
                                                                                      ,dataFrame.index[-1]))
                    dataFrame.iloc[maxHeaders:maxRows] = NaN
                    dataFrame = dataFrame.dropna(axis=0).copy()
        return dataFrame


    def _fill_year_standardize(self,year):
        if isinstance(year,str) == True:
           if re.search('\\d{4}$',year) is not None:
               # 针对恒瑞医药2014年报,普通股现金分红情况表,起分红年度为:2014,2013,2012,需要补充为2014年,2013年,2012年
               #if re.search('年',year) is None:
               year = re.sub("(\\d{4})$","\g<1>年",year)
           elif re.search('\\d{4}[.]\\d+[.]\\d{2}',year) is not None:
               # 国金证券2016年报,丽珠集团：2014年年度报告, 合并资产负债表的表头出现 : 2015.12.31  2014.12.31
               year = re.sub("(\\d{4})[.]\\d+[.]\\d{2}","\g<1>年",year)
           elif re.search('\\d{4}[-]\\d+[-]\\d{2}',year) is not None:
               # 尚荣医疗2015年半年度报告, 合并资产负债表,表头出现: 2015-6-30 2014-12-31
               year = re.sub("(\\d{4})[-]\\d+[-]\\d{2}","\g<1>年",year)
        return year


    def _get_max_headers(self,tableName):
        maxHeaders = self.dictTables[tableName]['maxHeaders']
        fieldFromHeader = self.dictTables[tableName]['fieldFromHeader']
        if len(fieldFromHeader) != 0 :
            #对于从表头用于字段的表,如分季度主要财务数据,无形资产情况,合并所有者权益变动表, maxHeaders原样返回
            return maxHeaders
        reportType = self.standard._get_report_type_by_filename(self.gConfig['sourcefile'])
        if reportType != '年度报告':
            # 对于第一季度报告,第三季度报告,半年读报告,设置maxHeaders = 2,即只保留一行数据, 即当年数据,往年数据则去掉,因为第二行数据各财报的定义不一致
            maxHeaders = 2
        return maxHeaders


    def _major_accounting_data_tailor(self,dataFrame,tableName):
        #针对第三季度报告中的主要会计数据进行裁剪,去掉第二个标题头之前的所有行数据,因为这几行数据不规范
        if dataFrame.shape[0] == 0:
            return dataFrame
        isSecondHeaderRowNext = False
        if self._is_third_quarter_accounting_data(tableName):
            for index,field in enumerate(dataFrame.iloc[:,0]):
                isHeaderInRow = self._is_header_in_row(dataFrame.iloc[index].tolist(), tableName)
                isRowAllInvalid = self._is_row_all_invalid_exclude_blank(dataFrame.iloc[index])
                if isRowAllInvalid == False:
                    if isHeaderInRow == False:
                        isSecondHeaderRowNext = True
                    elif isSecondHeaderRowNext == True:
                        break
                dataFrame.iloc[index] = NaN
        dataFrame = dataFrame.dropna(axis=0).copy()
        return dataFrame


    def _is_third_quarter_accounting_data(self,tableName):
        #如果是第三季度报告,主要会计数据,则返回Ture,这个数据需要特殊处理
        reportType = self.standard._get_report_type_by_filename(self.gConfig['sourcefile'])
        isThirdQuarterAccountingData = (reportType == '第三季度报告' and tableName == '主要会计数据')
        return isThirdQuarterAccountingData


    def _is_third_quarter_profit_data(self,tableName):
        #如果是第三季度报告,合并利润表,则返回Ture,这个数据需要特殊处理
        reportType = self.standard._get_report_type_by_filename(self.gConfig['sourcefile'])
        isThirdQuarterAccountingData = (reportType == '第三季度报告' and tableName == '合并利润表')
        return isThirdQuarterAccountingData


    def _rowDiscard(self, row, tableName):
        if self._is_header_in_row(row.tolist(), tableName):  # and row[0] == NULLSTR:
            row = row.apply(lambda value: NaN)
        return row


    def _discard_header_row(self, dataFrame, tableName):
        # 针对主要会计数据,去掉其表头
        dataFrame = dataFrame.apply(lambda row: self._rowDiscard(row, tableName), axis=1)
        dataFrame = dataFrame.dropna(axis=0).copy()
        return dataFrame


    def _get_field_len(self,field):
        if self._is_valid(field):
            fieldLen = len(field)
        else:
            # 如果是None和NULLSTR,这两种情况是不占用字段长度的,返回0
            fieldLen = 0
        return fieldLen


    def _get_aliased_fields(self, fieldList,tokenName, tableName):
        assert fieldList is not None, 'sourceRow(%s) must not be None' % fieldList
        aliasedFields = fieldList
        fieldAlias = self.dictTables[tableName][tokenName + 'Alias']
        fieldAliasKeys = list(self.dictTables[tableName][tokenName + 'Alias'].keys())
        if len(fieldAliasKeys) > 0:
            if isinstance(fieldList,list):
                aliasedFields = [utile.alias(field, fieldAlias) for field in fieldList]
            else:
                aliasedFields = utile.alias(fieldList, fieldAlias)
        return aliasedFields


    def _get_merged_row(self, sourceRow, mergeRow, isFieldJoin=False):
        #当isHorizontalTable=True时,为转置表,如普通股现金分红情况表,这个时候是对字段合并,采用字段拼接方式,其他情况采用替换方式
        mergedRow = [self._merge(field1, field2, isFieldJoin) for field1, field2 in zip(mergeRow, sourceRow)]
        return mergedRow


    def _get_duplicated_field(self,fieldList):
        dictFieldDuplicate = dict(zip(fieldList,[0]*len(fieldList)))
        def duplicate(fieldName):
            dictFieldDuplicate.update({fieldName:dictFieldDuplicate[fieldName] + 1})
            if dictFieldDuplicate[fieldName] > 1:
                fieldName += str(dictFieldDuplicate[fieldName] - 1)
            return fieldName
        duplicatedField = [duplicate(fieldName) for fieldName in fieldList]
        return duplicatedField


    def _is_row_all_invalid_exclude_blank(self,row:DataFrame):
        #如果该行以None开头,其他所有字段都是None或NULLSTR,则返回True
        isRowAllInvalid = False
        if (row == NULLSTR).all():
            #如果是空行,返回False,空行有特殊用途,一般加到最后一行来驱动前一个字段的合并
            return isRowAllInvalid
        mergedField = reduce(self._merge,row.tolist())
        #解决上峰水泥2017年中出现" ,None,None,None,None,None"的情况,以及其他年报中出现"None,,None,,None"的情况.
        isRowAllInvalid = not self._is_valid(mergedField)
        return isRowAllInvalid


    def _is_row_not_any_none(self,row:DataFrame,tableName):
        """
        args:
            row - dataFrame数据中的一行
            tableName - 表名, 用于从interpreterAccounting.Json中获取参数的索引
        return:
            isRowNotAnyNone - True or False, True 表示在本行中没有任何一个None. 判断原则:
            1) 如果是合并所有者权益变动表,数据的起始行的第一个字段为None,需要去掉, 否则会误判. 去掉后判断剩下的字段,如果没有一个None,返回True,否则返回False
            2) 对于其他表,只要一行中出现一个None,返回False
        """
        isRowNotAnyNone = False
        if len(row.values) <= 1:
            return isRowNotAnyNone
        # 解决 天山股份：2018年半年度报告,合并所有者权益变动表, 数据起始行第一个字段为None, 需要把第一个字段去掉: None 1,048,722 4,054,364,3	180,102,29	39,200,830.	271,961,81	1,756,703,4	915,452,04	8,266,507,7
        if tableName == '合并所有者权益变动表':
            cutedRow = row[1:]
        else:
            cutedRow = row
        isRowNotAnyNone = (cutedRow != NONESTR).all()
        return isRowNotAnyNone


    def _create_tables(self):
        for reportType in self.standard.reportTypes:
            tableNames = self.dictReportType[reportType]
            tablePrefix = self.standard._get_tableprefix_by_report_type(reportType)
            self._create_tables_by_type(tablePrefix,tableNames)


    def _create_tables_by_type(self, tablePrefix,tableNames):
        #用于向Sqlite3数据库中创建新表
        conn = self.database._get_connect()
        cursor = conn.cursor()
        allTables = self.database._fetch_all_tables(cursor)
        allTables = list(map(lambda x:x[0],allTables))
        for tableName in tableNames:
            targetTableName = tablePrefix + tableName
            if targetTableName not in allTables:
                sql = " CREATE TABLE IF NOT EXISTS [%s] ( \n\t\t\t\t\t" % targetTableName
                for commonFiled, type in self.commonFields.items():
                    sql = sql + "[%s] %s\n\t\t\t\t\t," % (commonFiled, type)
                #由表头转换生产的字段
                fieldFromHeader = self.dictTables[tableName]["fieldFromHeader"]
                if len(fieldFromHeader) != 0:
                    for field in fieldFromHeader:
                        sql = sql + "[%s] VARCHAR(20)\n\t\t\t\t\t,"%field
                sql = sql[:-1]  # 去掉最后一个逗号
                #创建新表
                standardizedFields = self.dictTables[tableName]['fieldName']
                duplicatedFields = self._get_duplicated_field(standardizedFields)
                for fieldName in duplicatedFields:
                    if fieldName is not NaN:
                        sql = sql + "\n\t\t\t\t\t,[%s]  NUMERIC"%fieldName
                sql = sql + '\n\t\t\t\t\t)'
                try:
                    conn.execute(sql)
                    conn.commit()
                    print('创建数据库表%s成功' % (targetTableName))
                except Exception as e:
                    # 回滚
                    conn.rollback()
                    print(e,' 创建数据库表%s失败' % targetTableName)

                #创建索引
                sql = "CREATE INDEX IF NOT EXISTS [%s索引] on [%s] (\n\t\t\t\t\t"%(targetTableName,targetTableName)
                sql = sql + ", ".join(str(field) for field,value in self.commonFields.items()
                                     if value.find('NOT NULL') >= 0)
                sql = sql + '\n\t\t\t\t\t)'
                try:
                    conn.execute(sql)
                    conn.commit()
                    print('创建数据库%s索引成功' % (targetTableName))
                except Exception as e:
                    # 回滚
                    conn.rollback()
                    print(e,' 创建数据库%s索引失败' % targetTableName)
        cursor.close()
        conn.close()


    def initialize(self):
        self.loggingspace.clear_directory(self.loggingspace.directory)


def create_object(gConfig):
    parser = DocParserSql(gConfig)
    parser.initialize()
    return parser