#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 12/9/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserSql.py
# @Note    : 用于sql数据库的读写

from interpreterAccounting.docparser.docParserBaseClass import  *
import sqlite3 as sqlite
#from sqlalchemy import create_engine
import pandas as pd
from pandas import DataFrame
from ply import lex
import time


class DocParserSql(DocParserBase):
    def __init__(self,gConfig):
        super(DocParserSql, self).__init__(gConfig)
        self._create_tables()
        self.process_info = {}
        self.dataTable = {}
        self.checkpointIsOn = self.gConfig['checkpointIsOn'.lower()]
        self.dictLexers = self._construct_lexers()


    def _construct_lexers(self):
        dictLexer = dict()
        for tableName in self.dictTables.keys():
            dictLexer.update({tableName:dict()})
            for tokenName in ['field','header']:
                #构建field lexer 和 header lexer
                dictTokens = self._get_dict_tokens(tableName,tokenName=tokenName)
                lexer = self._get_lexer(dictTokens)
                dictLexer[tableName].update({'lexer'+tokenName.title(): lexer, "dictToken" + tokenName.title(): dictTokens})
                self.logger.info('success to create %s lexer for %s!' % (tokenName,tableName))
            #构建header lexer
            #dictTokens = self._get_dict_tokens(tableName,tokenName='header')
            #lexer = self._get_lexer(dictTokens)
            #dictLexer[tableName].update({'lexerHeader': lexer, "dictTokenHeader": dictTokens})
            #self.logger.info('success to create field lexer for %s!' % tableName)
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
            # t.type = 'ILLEGALFIELD'
            # return t

        # Build the lexer
        lexer = lex.lex(outputdir=self.working_directory)
        return lexer


    def _get_dict_tokens(self, tableName,tokenName='field'):
        # 对所有的表字段进行标准化
        #standardFields = self._get_standardized_field(self.dictTables[tableName][tokenName + 'Name'], tableName)
        #standardFields = self._get_standardized_keyword(self.dictTables[tableName][tokenName + 'Name']
        #                                                , self.dictTables[tableName][tokenName + 'Standardize'])
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
        '''dictAlias = {}
        for key, value in self.dictTables[tableName][tokenName+'Alias'].items():
            #keyStandard = self._get_standardized_field(key, tableName)
            #valueStandard = self._get_standardized_field(value, tableName)
            keyStandard = self._get_standardized_keyword(key, self.dictTables[tableName][tokenName+'Standardize'])
            valueStandard = self._get_standardized_keyword(value, self.dictTables[tableName][tokenName+'Standardize'])
            if valueStandard == passMatching:
                #把PASSMATCHING加入到dictTocken中
                fieldsIndex.update({'PASSMATCHING': passMatching})
                dictAlias.update({key: passMatching})
            elif keyStandard != valueStandard:
                dictAlias.update({keyStandard: valueStandard})
            else:
                self.logger.warning("%s has same field after standardize in fieldAlias:%s %s" % (tableName, key, value))
        # 判断fieldAlias中经过标准化后是否有重复字段,如果存在,则配置是不合适的
        if len(self.dictTables[tableName][tokenName+'Alias'].keys()) != len(dictAlias.keys()):
            self.logger.warning('It is same field after standard in fieldAlias of %s:%s'
                                % (tableName, ' '.join(self.dictTables[tableName][tokenName+'Alias'].keys())))'''
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
        # dictTokens.update({'VIRTUALFIELD':self.gJsonInterpreter['VIRTUALFIELD']})
        #discardField = self.dictTables[tableName][tokenName+'Discard']
        #standardDiscardFields = self._get_standardized_field(discardField, tableName)
        #standardDiscardFields = self._get_standardized_keyword(discardField, self.dictTables[tableName][tokenName+'Standardize'])
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
                    "%s has dupicated (fieldDiscard/headerDiscard) with (fieldName/headerName):%s!" % (tableName, ' '.join(list(fieldJoint))))
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


    def loginfo(text = 'running '):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self,*args, **kwargs):
                result = func(self,*args, **kwargs)
                resultForLog = result
                columns = 0
                if isinstance(result,tuple):
                    resultForLog = result[0].T.copy()
                    columns = result[0].iloc[0]
                self.logger.info('%s %s() \n\t%s,%s%s,\t%s:\n\t%s\n\t%s\n\t columns=%s'
                                  % (text, func.__name__,
                                     self.dataTable['公司名称'],self.dataTable['报告时间'],self.dataTable['报告类型'],
                                     args[-1],'',
                                     resultForLog,columns))
                return result
            return wrapper
        return decorator


    def writeToStore(self, dictTable):
        self.dataTable = dictTable
        table = dictTable['table']
        tableName = dictTable['tableName']
        if len(table) == 0:
            self.logger.error('failed to process %s, the table is empty'%tableName)
            return

        #if dictTable['tableEnd'] == True:
        self.process_info.update({tableName:{'processtime':time.time()}})
        dataframe,countTotalFields = self._table_to_dataframe(table,tableName)#pd.DataFrame(table[1:],columns=table[0],index=None)

        #对数据进行预处理,两行并在一行的分开,去掉空格等
        dataframe = self._process_value_pretreat(dataframe,tableName)

        #针对合并所有者权益表的前三列空表头进行合并,对转置表进行预转置,使得其处理和其他表一致
        dataframe = self._process_header_merge_pretreat(dataframe, tableName)

        #把跨多个单元格的表字段名合并成一个
        dataframe = self._process_field_merge_simple(dataframe,'field',tableName)
        #dataframe = self._process_field_merge_simple_expired(dataframe, tableName)

        #去掉空字段及无用字段
        dataframe = self._process_field_discard(dataframe,'field' ,tableName)

        #同一张表的相同字段在不同财务报表中名字不同, 需要统一为相同名称, 统一后再去重
        dataframe = self._process_field_alias(dataframe,'field', tableName)

        #对表进行转置,然后把跨多行的表头字段进行合并
        dataframe = self._process_header_merge_simple(dataframe,tableName)

        #去掉无用的表头;同时对水平表进行转置,把字段名由index转为column
        dataframe = self._process_header_discard(dataframe, tableName)

        # 同一张表的相同表头在不同财务报表中名字不同, 需要统一为相同名称, 统一后再去重
        dataframe = self._process_header_alias(dataframe, tableName)
        #把表头进行标准化
        #dataframe = self._process_header_standardize(dataframe,tableName)

        #去掉不必要的行数据,比如合并资产负债表有三行数据,最后一行往往是上期数据的修正,是不必要的
        dataframe = self._process_row_tailor(dataframe, tableName)

        #如果dataframe之有一行数据，说明从docParserPdf推送过来的数据是不正确的：只有表头，没有任何有效数据。
        if dataframe.shape[0] <= 1 or dataframe.shape[1] <= 1:
            self.logger.error("failed to process %s, the table has no data:%s,shape %s"%(tableName,dataframe.values,dataframe.shape))
            self.process_info.pop(tableName)
            return

        #同一张表的相同字段在不同财务报表中名字不同,需要统一为相同名称,统一后再去重
        #dataframe = self._process_field_alias(dataframe,tableName)

        #处理重复字段
        dataframe = self._process_field_duplicate(dataframe,tableName)

        #dataframe前面插入公共字段
        dataframe = self._process_field_common(dataframe, dictTable, countTotalFields,tableName)

        dataframe = self._process_value_standardize(dataframe,tableName)

        #把dataframe写入sqlite3数据库
        self._write_to_sqlite3(dataframe,tableName)
        #if dictTable['tableEnd'] == True:
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


    def _write_to_sqlite3(self, dataFrame:DataFrame,tableName):
        conn = self._get_connect()
        dataFrame = dataFrame.T
        sql_df = dataFrame.set_index(dataFrame.columns[0],inplace=False).T
        isRecordExist = self._is_record_exist(conn, tableName, sql_df)
        if isRecordExist:
            condition = self._get_condition(sql_df)
            sql = ''
            sql = sql + 'delete from {}'.format(tableName)
            sql = sql + '\nwhere ' + condition
            self._sql_executer(sql)
            self.logger.info("delete from {} where is {} {} {}!".format(tableName,sql_df['公司简称'].values[0]
                                                                        ,sql_df['报告时间'].values[0],sql_df['报告类型'].values[0]))
            sql_df.to_sql(name=tableName, con=conn, if_exists='append', index=False)
            conn.commit()
            self.logger.info("insert into {} where is {} {} {}!".format(tableName, sql_df['公司简称'].values[0]
                                                                        ,sql_df['报告时间'].values[0],sql_df['报告类型'].values[0]))
        else:
            sql_df.to_sql(name=tableName,con=conn,if_exists='append',index=False)
            conn.commit()
            self.logger.info("insert into {} where is {} {} {}!".format(tableName, sql_df['公司简称'].values[0]
                                                                            ,sql_df['报告时间'].values[0],sql_df['报告类型'].values[0]))
        conn.close()


    def _rowPretreat(self,row):
        self.lastValue = None
        row = row.apply(self._valuePretreat)
        return row


    def _valuePretreat(self,value):
        try:
            if isinstance(value,str):
                if value != NONESTR and value != NULLSTR:
                    value = re.sub('不适用$',NULLSTR,value)
                    value = re.sub('^附注',NULLSTR,value)#解决隆基股份2017年报中,合并资产负债表中的出现"附注六、1"
                    value = re.sub('元$',NULLSTR,value)#解决海螺水泥2018年报中,普通股现金分红情况表中出现中文字符,导致_process_field_merge出错
                    value = re.sub('^上升',NULLSTR,value)#解决京新药业2017年年报,主要会计数据的一列数据中出现"上升 0.49个百分点"
                    value = re.sub('个百分点$',NULLSTR,value)#解决京新药业2017年年报,海螺水泥2018年年报主要会计数据的一列数据中出现"上升 0.49个百分点"
                    value = re.sub('^注释',NULLSTR,value)#解决灰顶科技2019年年报中,合并资产负债表,合并利润表中的字段出现'注释'
                    value = re.sub('^）\\s*',NULLSTR,value)
                    value = re.sub('^同比增加',NULLSTR,value)#解决冀东水泥：2017年年度报告.PDF主要会计数据中的一列数据中出现'同比增加',导致_precess_header_merge_simple误判
                    value = re.sub('注$', NULLSTR, value)#解决康泰生物2018年年报中普通股现金分红情况表中出现中文字符'注',导致_process_merge_header_simple出问题
                    value = re.sub('^增加', NULLSTR, value) #解决海天味业2014年报中主营业物质的数据中出现,'增加','减少'
                    value = re.sub('^减少', NULLSTR, value) #解决海天味业2014年报中主营业物质的数据中出现,'增加','减少'
                    value = re.sub('^下降', NULLSTR, value)  # 解决海螺水泥2014年报中主营业物质的数据中出现,'下降'
                    value = re.sub('(^[一二三四五六七八九十〇]{1,2})、', NULLSTR, value)
                    result = re.split("[ ]{2,}",value,maxsplit=1)
                    if len(result) > 1:
                        value,self.lastValue = result
                else:
                    if self.lastValue != None and value == NONESTR:
                        value = self.lastValue
                        self.lastValue = None
        except Exception as e:
            print(e)
        return value


    def _process_value_pretreat(self,dataFrame:DataFrame,tableName):
        #采用正则表达式替换空字符,对一个字段中包含两个数字字符串的进行拆分
        #解决奥美医疗2018年年报,主要会计数据中,存在两列数值并列到了一列,同时后接一个None的场景.
        #东材科技2018年年报,普通股现金分红流量表,表头有很多空格,影响_process_header_discard,需要去掉
        dataFrame.iloc[1:,1:]  = dataFrame.iloc[1:,1:].apply(self._rowPretreat,axis=1)
        #dataFrame.iloc[:,:] = dataFrame.iloc[:,:].apply(lambda row:row.apply(lambda x: x.replace(' ',NULLSTR).replace('(','（').replace(')','）')))
        dataFrame.iloc[:,:] = dataFrame.iloc[:,:].apply(lambda row:row.apply(self._replace_value))
        return dataFrame


    def _process_header_merge_pretreat(self, dataFrame, tableName):
        isHorizontalTable = self.dictTables[tableName]['horizontalTable']
        mergedRow = None
        lastIndex = 0
        # 增加blankFrame来驱动最后一个field的合并
        blankFrame = pd.DataFrame([''] * len(dataFrame.columns.values), index=dataFrame.columns).T
        dataFrame = dataFrame.append(blankFrame)
        while self._is_row_all_invalid(dataFrame.iloc[0]):
            # 如果第一行数据全部为无效的,则删除掉. 解决康泰生物：2016年年度报告.PDF,合并所有者权益变动表中第一行为全None行,导致标题头不对的情况
            # 但是解析出的合并所有者权益变动表仍然是不对的,原因是合并所有者权益变动表第二页的数据被拆成了两张无效表,而用母公司合并所有者权益变动表的数据填充了.
            dataFrame.iloc[0] = NaN
            dataFrame = dataFrame.dropna()
        for index, field in enumerate(dataFrame.iloc[:, 0]):
            #isRowNotAnyNone = self._is_row_not_any_none(dataFrame.iloc[index])
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
                    else:
                        mergedRow = None
                    #elif isHorizontalTable == True :
                    #    if isRowNotAnyNone == True:
                    #        if self._is_first_field_in_row(mergedRow, tableName):
                    #            if index > lastIndex + 1:
                    #                dataFrame.iloc[lastIndex] = mergedRow
                    #                dataFrame.iloc[lastIndex + 1:index] = NaN
                    #            mergedRow = None
                    #    else:
                    #        #康泰生物2019年年报普通股现金分红情况表,有一列全部为None,此时isRowNotAnyNone失效
                    #        if self._is_first_field_in_row(mergedRow,tableName) and self._is_first_field_in_row(field,tableName):
                    #            if index > lastIndex + 1:
                    #                dataFrame.iloc[lastIndex] = mergedRow
                    #                dataFrame.iloc[lastIndex + 1:index] = NaN
                    #            mergedRow = None
                    #            self.logger.warning('%s 出现某一列全部解析为None情况,just for debug!'%tableName)
                    #else:
                    #    mergedRow = None
                else:
                    if isHeaderInMergedRow == False:
                        #解决再升科技2018年年报,合并所有者权益变动表在每个分页中插入了表头
                        # 解决大立科技：2018年年度报告,有一行", , , ,调整前,调整后, , , , ",满足isRowNotAnyNone==True条件,但是需要继续合并
                        mergedRow = None
            if mergedRow is None:
                mergedRow = dataFrame.iloc[index].tolist()
                lastIndex = index
            else:
                mergedRow = self._get_merged_row(dataFrame.iloc[index].tolist(), mergedRow, isFieldJoin=True)

        if isHorizontalTable == True:
            dataFrame = dataFrame.dropna(axis=0)
            #把第一列做成索引
            dataFrame.set_index(0,inplace=True)
            dataFrame = dataFrame.T.copy()
        else:
            dataFrame.columns = dataFrame.iloc[0].copy()
            #主要会计数据的表头通过上述过程去不掉,采用专门的函数去掉
            dataFrame = self._discard_header_row(dataFrame,tableName)
        return dataFrame

    '''
    def _process_header_merge_pretreat_outdate(self, dataFrame, tableName):
        isHorizontalTable = self.dictTables[tableName]['horizontalTable']
        mergedRow = None
        #firstHeader = self.dictTables[tableName]['header'][0]
        lastIndex = 0
        # 增加blankFrame来驱动最后一个field的合并
        blankFrame = pd.DataFrame([''] * len(dataFrame.columns.values), index=dataFrame.columns).T
        dataFrame = dataFrame.append(blankFrame)
        #isRowAllInvalid = self._is_row_all_invalid(dataFrame.iloc[0]) #里面有一个逻辑:如果所有都是空行,则返回False,不再适用
        #isRowAllInvalid = (dataFrame.iloc[0] == NULLSTR).all()
        #康泰生物：2016年年度报告.PDF合并所有者权益变动表第一行全部为None
        #isRowAllInvalid = self._is_row_all_blank(dataFrame.iloc[0]) or self._is_row_all_none(dataFrame.iloc[0])
        while self._is_row_all_invalid(dataFrame.iloc[0]):
            # 如果第一行数据全部为无效的,则删除掉. 解决康泰生物：2016年年度报告.PDF,合并所有者权益变动表中第一行为全None行,导致标题头不对的情况
            # 但是解析出的合并所有者权益变动表仍然是不对的,原因是合并所有者权益变动表第二页的数据被拆成了两张无效表,而用母公司合并所有者权益变动表的数据填充了.
            dataFrame.iloc[0] = NaN
            dataFrame = dataFrame.dropna()
        for index, field in enumerate(dataFrame.iloc[:, 0]):
            #isRowNotAnyNone = self._is_row_not_any_none(dataFrame.iloc[index])
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
                    else:
                        mergedRow = None
                    ''
                    elif isHorizontalTable == True :
                        if isRowNotAnyNone == True:
                            if self._is_first_field_in_row(mergedRow, tableName):
                                if index > lastIndex + 1:
                                    dataFrame.iloc[lastIndex] = mergedRow
                                    dataFrame.iloc[lastIndex + 1:index] = NaN
                                mergedRow = None
                        else:
                            #康泰生物2019年年报普通股现金分红情况表,有一列全部为None,此时isRowNotAnyNone失效
                            if self._is_first_field_in_row(mergedRow,tableName) and self._is_first_field_in_row(field,tableName):
                                if index > lastIndex + 1:
                                    dataFrame.iloc[lastIndex] = mergedRow
                                    dataFrame.iloc[lastIndex + 1:index] = NaN
                                mergedRow = None
                                self.logger.warning('%s 出现某一列全部解析为None情况,just for debug!'%tableName)
                    else:
                        mergedRow = None''
                else:
                    if isHeaderInMergedRow == False:
                        #解决再升科技2018年年报,合并所有者权益变动表在每个分页中插入了表头
                        # 解决大立科技：2018年年度报告,有一行", , , ,调整前,调整后, , , , ",满足isRowNotAnyNone==True条件,但是需要继续合并
                        mergedRow = None
                #if isRowNotAnyNone == True and isHorizontalTable == True: #and isHeaderInRow == True:
                    #表头或表字段所在的起始行,针对普通股分红情况表，在该函数中要把字段的合并也一起做了
                #    if self._is_first_field_in_row(mergedRow, tableName):
                #        if index > lastIndex + 1:
                #            dataFrame.iloc[lastIndex] = mergedRow
                #            dataFrame.iloc[lastIndex + 1:index] = NaN
                #        mergedRow = None
                #if not (isHeaderInRow == True and isHeaderInMergedRow == True):
                    #解决大立科技：2018年年度报告,有一行", , , ,调整前,调整后, , , , ",满足isRowNotAnyNone==True条件,但是需要继续合并
                #    mergedRow = None
            if mergedRow is None:
                mergedRow = dataFrame.iloc[index].tolist()
                lastIndex = index
            else:
                mergedRow = self._get_merged_row(dataFrame.iloc[index].tolist(), mergedRow, isFieldJoin=True)

        if isHorizontalTable == True:
            #如果是转置表,则在此处做一次转置,后续的处理就和非转置表保持一致了
            #去掉最后一行空行
            #indexDiscardField = dataFrame.iloc[:, 0].isin(self._get_invalid_field())
            #dataFrame.loc[indexDiscardField] = NaN
            #解决（300326）凯利泰：2018年年度报告.PDF，普通股现金分红请表，除了第一行解析正常为“”
            ''
            #这段代码被_process_header_merge_simple取代
            indexDiscardField = [not self._is_first_field_in_row(x,tableName) for x in dataFrame.iloc[:,0]]
            indexDiscardField[0] = False
            dataFrame.loc[indexDiscardField] = NaN''
            dataFrame = dataFrame.dropna(axis=0)
            #把第一列做成索引
            dataFrame.set_index(0,inplace=True)
            dataFrame = dataFrame.T.copy()
        else:
            dataFrame.columns = dataFrame.iloc[0].copy()
            #主要会计数据的表头通过上述过程去不掉,采用专门的函数去掉
            dataFrame = self._discard_header_row(dataFrame,tableName)
        return dataFrame
    '''

    @loginfo()
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
        #dictFieldPos = dict({0:{"lexpos":0,'value':NULLSTR,'type':NULLSTR}})
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
            isRowNotAnyNone = self._is_row_not_any_none(dataFrame.iloc[index])
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
                    mergedRow = None
                    #开始搜寻下一个字段
                    posIndex = posIndex + 1
                else:
                    #正对字段名是None和NULLSTR的处理,这两种情况下,该字段不占用字符串长度
                    if field == NULLSTR and isRowNotAnyNone:
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
        firstColumn = dataFrame.columns.values[0]
        dataFrame.set_index(firstColumn,inplace=True)
        dataFrame = dataFrame.T.reset_index()
        dataFrame.columns.values[0] = firstColumn

        isHorizontalTable = self.dictTables[tableName]['horizontalTable']
        if isHorizontalTable == False:
            #对于非转置表,在_process_header_merge_pretreat中实际对header字段做了合并,除非对字段做标准化,否则暂时不需要再次进行合并
            #return dataFrame
            pass
        dataFrame = self._process_field_merge_simple(dataFrame,"header",tableName)
        #dataFrame = dataFrame.T.reset_index().T
        #dataFrame.set_index(dataFrame.columns[0],inplace=True)
        #dataFrame = dataFrame.T.copy()
        return dataFrame


    @loginfo()
    def _process_field_common(self, dataFrame, dictTable, countFieldDiscard,tableName):
        #在dataFrame前面插入公共字段
        fieldFromHeader = self.dictTables[tableName]["fieldFromHeader"]
        countColumns = len(dataFrame.columns) + countFieldDiscard
        index = 0
        for (commonFiled, _) in self.commonFileds.items():
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
                    #高德红外2018年报,无效值用'--'填充,部分年报无效值用'-'填充
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


    def _process_field_duplicate(self,dataFrame,tableName):
        # 重复字段处理,放在字段标准化之后
        duplicatedFields = self._get_duplicated_field(dataFrame.iloc[0].tolist())

        #standardizedFields = self._get_standardized_field(self.dictTables[tableName]['fieldName'], tableName)
        standardizedFields = self.dictTables[tableName]['fieldName']
        duplicatedFieldsStandard = self._get_duplicated_field(standardizedFields)

        dataFrame.iloc[0] = duplicatedFields
        duplicatedFieldsResult = []
        for field in duplicatedFields:
            if field in duplicatedFieldsStandard:
                duplicatedFieldsResult += [field]
            else:
                self.logger.warning('failed to add field: %s is not exist in %s'%(field,tableName))
                #删除该字段
                indexDiscardField = dataFrame.iloc[0].isin([field])
                discardColumns = indexDiscardField[indexDiscardField == True].index.tolist()
                #dataFrame.T.loc[indexDiscardField] = NaN
                dataFrame[discardColumns] = NaN
                dataFrame = dataFrame.dropna(axis=1).copy()
        return dataFrame


    def _process_header_discard(self, dataFrame, tableName):
        '''
        #去掉无用的表头;把字段名由index转为column
        headerDiscard = self.dictTables[tableName]['headerDiscard']
        headerDiscardPattern = '|'.join(headerDiscard)
        #同时对水平表进行转置,把字段名由index转为column,便于插入sqlite3数据库
        #dataFrame = dataFrame.T.copy()
        #删除需要丢弃的表头,该表头由self.dictTables[tableName]['headerDiscard']定义
        indexDiscardHeader = [self._is_field_matched(headerDiscardPattern,x) for x in dataFrame.index.values]
        #主要会计数据的第一个index为空字段,该代码会判断为Ture,所以要重新设置为False
        indexDiscardHeader[0] = False
        dataFrame.loc[indexDiscardHeader] = NaN
        indexDiscardHeader = dataFrame.index.isin(self._get_invalid_field())
        indexDiscardHeader[0] = False
        dataFrame.loc[indexDiscardHeader] = NaN
        dataFrame = dataFrame.dropna(axis=0).copy()'''
        dataFrame = self._process_field_discard(dataFrame,'header',tableName)
        return dataFrame


    #@loginfo()
    def _process_field_discard(self, dataFrame, tokenName,tableName):
        #去掉空字段,针对主要会计数据这张表,需要提出掉其空字段
        #对于普通股现金分红情况表,则忽略这一过程
        fieldDiscard = self.dictTables[tableName][tokenName + 'Discard']
        fieldDiscardPattern = '|'.join(fieldDiscard)
        indexDiscardField = [self._is_field_matched(fieldDiscardPattern, x) for x in dataFrame.iloc[:,0]]
        # 主要会计数据的第一个index为空字段,该代码会判断为Ture,所以要重新设置为False
        #indexDiscardHeader[0] = False
        dataFrame.loc[indexDiscardField] = NaN
        #对非中文字符全部替换为中文字符,这部分工作已经在_fields_replace_punctuate完成
        #fieldDiscard = [field.replace('(','（').replace(')','）').replace(' ',NULLSTR) for field in fieldDiscard]
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
        #aliasedFields = self._get_aliased_fields(dataFrame.iloc[:,0].tolist(),tokenName, tableName)
        #dataFrame.iloc[0] = aliasedFields
        #对于合并利润表,需要在标准化后进行统一命名
        #dataFrame = self._process_field_standardize(dataFrame,tokenName,tableName)
        standardizedFields = self._get_standardized_keyword(dataFrame.iloc[:,0].tolist(),self.dictTables[tableName][tokenName+'Standardize'])
        dataFrame.iloc[:,0] = standardizedFields
        aliasedFields = self._get_aliased_fields(dataFrame.iloc[:,0].tolist(),tokenName, tableName)
        dataFrame.iloc[:,0] = aliasedFields
        if tokenName == 'header':
            # 对columns[0]做特殊处理
            column0 = dataFrame.columns.values[0]
            standardizedColumns0 = self._get_standardized_keyword(column0,self.dictTables[tableName][tokenName + 'Standardize'])
            aliasedColumns0 = self._get_aliased_fields(standardizedColumns0, tokenName, tableName)
            if aliasedColumns0 != NaN:
                dataFrame.columns.values[0] = aliasedColumns0
        #统一命名之后,再次进行标准化
        #dataFrame = self._process_field_standardize(dataFrame, tableName)
        dataFrame = dataFrame.dropna(axis = 0).copy()
        return dataFrame

    '''
    def _process_field_alias_outdate(self, dataFrame, tableName):
        #同一张表的相同字段在不同财务报表中名字不同,需要统一为相同名称
        #针对主要会计数据,需要在标准化前进行统一命名
        aliasedFields = self._get_aliased_fields(dataFrame.iloc[0].tolist(), tableName)
        dataFrame.iloc[0] = aliasedFields
        #对于合并利润表,需要在标准化后进行统一命名
        dataFrame = self._process_field_standardize(dataFrame,tableName)
        aliasedFields = self._get_aliased_fields(dataFrame.iloc[0].tolist(), tableName)
        dataFrame.iloc[0] = aliasedFields
        #统一命名之后,再次进行标准化
        dataFrame = self._process_field_standardize(dataFrame, tableName)
        return dataFrame
   '''
    '''
    def _process_header_standardize(self,dataFrame,tableName):
        #把表头字段进行标准化
        standardizedHeaders = self._get_standardized_header(dataFrame.index.values[1:].tolist(),tableName)
        dataFrame.index.values[1:] = standardizedHeaders
        #在标准化后,某些无用字段可能被标准化为NaN,需要去掉
        #dataFrame.loc[NaN] = NaN
        #dataFrame = dataFrame.dropna(axis=1).copy()
        return dataFrame
   '''

    '''
    def _process_field_standardize(self,dataFrame,tokenName,tableName):
        #把表字段进行标准化,把所有的字段名提取为两种模式,如:利息收入,一、营业总收入
        #standardizedFields = self._get_standardized_field(dataFrame.iloc[0].tolist(),tableName)
        standardizedFields = self._get_standardized_keyword(dataFrame.iloc[:,0].tolist(),self.dictTables[tableName][tokenName+'Standardize'])
        dataFrame.iloc[:,0] = standardizedFields
        dataFrame = dataFrame.dropna(axis = 0).copy()
        return dataFrame
    '''

    def _process_row_tailor(self, dataFrame, tableName):
        #转化成需要的dataFrame
        dataFrame = dataFrame.T.reset_index().T
        dataFrame.set_index(dataFrame.columns[0],inplace=True)
        maxHeaders = self.dictTables[tableName]['maxHeaders']
        fieldFromHeader = self.dictTables[tableName]['fieldFromHeader']
        if len(fieldFromHeader) == 0:
            #对于合并所有者权益变动表,无形资产情况,分季度主要财务数据等表,不需要对多余的行进行裁剪
            maxRows = len(dataFrame.index.values)
            if maxRows > maxHeaders:
                #对于超出最大头长度的行进行拆解,解决华侨城A 2018年包中,合并资产负债表 有三列数据,其中最后一列数据是不需要的
                dataFrame.iloc[maxHeaders:maxRows] = NaN
                dataFrame = dataFrame.dropna(axis=0).copy()
                self.logger.warning("%s has %d row,only %s is needed!"%(tableName,maxRows,maxHeaders))
        return dataFrame


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


    #def _fieldname_pretreat(self,row):
    #    #将所有正则表达式中要用到的字符全部替换成中文字符
    #    row = row.apply(lambda value:value.replace('-', '－').replace('(','（').replace(')','）').replace('.','．'))
    #    return row


    def _get_aliased_fields(self, fieldList,tokenName, tableName):
        assert fieldList is not None, 'sourceRow(%s) must not be None' % fieldList
        aliasedFields = fieldList
        fieldAlias = self.dictTables[tableName][tokenName + 'Alias']
        fieldAliasKeys = list(self.dictTables[tableName][tokenName + 'Alias'].keys())
        if len(fieldAliasKeys) > 0:
            if isinstance(fieldList,list):
                aliasedFields = [self._alias(field, fieldAlias) for field in fieldList]
            else:
                aliasedFields = self._alias(fieldList,fieldAlias)
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

    '''
    def _get_standardized_field_strict(self,fieldList,tableName):
        assert fieldList is not None, 'sourceRow(%s) must not be None' % fieldList
        fieldStandardizeStrict = self.dictTables[tableName]['fieldStandardizeStrict']
        if isinstance(fieldList, list):
            standardizedFields = [self._standardize(fieldStandardizeStrict, field) for field in fieldList]
        else:
            standardizedFields = self._standardize(fieldStandardizeStrict, fieldList)
        return standardizedFields
    '''
    '''
    def _is_standardize_strict_mode(self,mergedFields, tableName):
        isStandardizeStrictMode = False
        standardizedFieldsStrict = self._get_standardized_field_strict(self.dictTables[tableName]['fieldName'],
                                                                       tableName)
        standardizedFieldsStrictPattern = '|'.join(standardizedFieldsStrict)
        if isinstance(standardizedFieldsStrictPattern, str) and isinstance(mergedFields, str):
            if standardizedFieldsStrictPattern != NULLSTR:
                matched = re.search(standardizedFieldsStrictPattern,mergedFields)
                if matched is not None:
                    isStandardizeStrictMode = True
        return isStandardizeStrictMode
    '''

    def _is_first_field_in_row(self, row_or_field, tableName):
        #对获取到的字段做标准化(需要的话),然后和配置表中代表最后一个字段(或模式)做匹配,如匹配到,则认为找到表尾
        #对于现金分红情况表,因为字段为时间,则用模式去匹配,匹配到一个即可认为找到表尾
        isFirstFieldInRow = False
        if row_or_field is None or row_or_field is NaN:
            return isFirstFieldInRow
        if isinstance(row_or_field, list):
            firstField = row_or_field[0]
        else:
            firstField = row_or_field
        if firstField == NULLSTR or firstField == NONESTR:
            return isFirstFieldInRow
        fieldFirst = self.dictTables[tableName]["fieldFirst"]
        #解决部分主要会计数据表中,前面几个字段拼成一起的情况
        fieldFirst = '^' + fieldFirst + '$'
        isFirstFieldInRow = self._is_field_matched(fieldFirst, firstField)
        return isFirstFieldInRow


    '''
    def _is_field_match_standardize(self, field, tableName):
        isFieldInList = False
        standardizedFields = self._get_standardized_field(self.dictTables[tableName]['fieldName'],tableName)
        aliasFields = list(self.dictTables[tableName]['fieldAlias'].keys())
        discardFields = list(self.dictTables[tableName]['fieldDiscard'])
        standardizedFields.extend(aliasFields)
        standardizedFields.extend(discardFields)
        standardizedFields = [field for field in standardizedFields if self._is_valid(field)]
        assert isinstance(standardizedFields,list),"patternList(%s) must be a list of string"%standardizedFields
        for pattern in standardizedFields:
            #pattern = '^' + pattern
            if self._is_field_matched(pattern,field):
                isFieldInList = True
                break
        return isFieldInList
    '''
    '''
    def _is_field_in_standardize_by_mode(self,field,isStandardizeStrict,tableName):
        if isStandardizeStrict == True:
            isFieldInStandardize = self._is_field_in_standardize_strict(field,tableName)
        else:
            isFieldInStandardize = self._is_field_in_standardize(field,tableName)
        return isFieldInStandardize
    '''
    '''
    def _is_field_in_standardize(self,field,tableName):
        # 把field按严格标准进行标准化,然后和判断该字段是否和同样方法标准化后的某个字段相同.
        isFieldInList = False
        standardizedFields = self._get_standardized_field(self.dictTables[tableName]['fieldName'],tableName)
        aliasFields = list(self.dictTables[tableName]['fieldAlias'].keys())
        discardFields = list(self.dictTables[tableName]['fieldDiscard'])
        standardizedFields.extend(aliasFields)
        standardizedFields.extend(discardFields)
        standardizedFields = [field for field in standardizedFields if self._is_valid(field)]
        assert isinstance(standardizedFields,
                          list), "patternList(%s) must be a list of string" % standardizedFields
        fieldStrict = self._get_standardized_field(field, tableName)
        if fieldStrict in standardizedFields:
            isFieldInList = True
        return isFieldInList
    '''
    '''
    def _is_field_in_standardize_strict(self, field,tableName):
        #把field按严格标准进行标准化,然后和判断该字段是否和同样方法标准化后的某个字段相同.
        isFieldInList = False
        standardizedFieldsStrict = self._get_standardized_field_strict(self.dictTables[tableName]['fieldName'],tableName)
        aliasFields = list(self.dictTables[tableName]['fieldAlias'].keys())
        discardFields = list(self.dictTables[tableName]['fieldDiscard'])
        standardizedFieldsStrict.extend(aliasFields)
        standardizedFieldsStrict.extend(discardFields)
        standardizedFieldsStrict = [field for field in standardizedFieldsStrict if self._is_valid(field)]
        assert isinstance(standardizedFieldsStrict,list),"patternList(%s) must be a list of string"%standardizedFieldsStrict
        fieldStrict = self._get_standardized_field_strict(field,tableName)
        if fieldStrict in standardizedFieldsStrict:
            isFieldInList = True
        return isFieldInList
    '''

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


    def _is_row_all_invalid(self,row:DataFrame):
        mergedField = reduce(self._merge,row.tolist())
        #解决上峰水泥2017年中出现" ,None,None,None,None,None"的情况,以及其他年报中出现"None,,None,,None"的情况.
        isRowAllInvalid = not self._is_valid(mergedField)
        return isRowAllInvalid


    def _is_row_not_any_none(self,row:DataFrame):
        return (row != NONESTR).all()


    '''
    def _is_row_all_blank(self,row:DataFrame):
        return (row == NULLSTR).all()


    def _is_row_all_none(self,row:DataFrame):
        return (row == NONESTR).all()
   '''

    def _is_record_exist(self, conn, tableName, dataFrame:DataFrame):
        #用于数据在插入数据库之前,通过组合的关键字段判断记录是否存在.
        #对于Sqlit3,字符串表示为'string' ,而不是"string".
        condition = self._get_condition(dataFrame)
        sql = 'select count(*) from {} where '.format(tableName) + condition
        result = conn.execute(sql).fetchall()
        isRecordExist = False
        if len(result) > 0:
            isRecordExist = (result[0][0] > 0)
        return isRecordExist


    def _get_condition(self,dataFrame):
        primaryKey = [key for key, value in self.commonFileds.items() if value.find('NOT NULL') >= 0]
        # 对于Sqlit3,字符串表示为'string' ,而不是"string".
        joined = list()
        for key in primaryKey:
            current = '(' + ' or '.join(['{} = \'{}\''.format(key,value) for value in set(dataFrame[key].tolist())]) + ')'
            joined = joined + list([current])
        condition = ' and '.join(joined)
        return condition


    def _get_connect(self):
        #用于获取数据库连接
        return sqlite.connect(self.database)


    def _fetch_all_tables(self, cursor):
        #获取数据库中所有的表,用于判断待新建的表是否已经存在
        try:
            cursor.execute("select name from sqlite_master where type='table' order by name")
        except Exception as e:
            print(e)
        return cursor.fetchall()


    def _create_tables(self):
        #用于向Sqlite3数据库中创建新表
        conn = self._get_connect()
        cursor = conn.cursor()
        allTables = self._fetch_all_tables(cursor)
        allTables = list(map(lambda x:x[0],allTables))
        for tableName in self.tableNames:
            if tableName not in allTables:
                sql = " CREATE TABLE IF NOT EXISTS [%s] ( \n\t\t\t\t\t" % tableName
                for commonFiled, type in self.commonFileds.items():
                    sql = sql + "[%s] %s\n\t\t\t\t\t," % (commonFiled, type)
                #由表头转换生产的字段
                fieldFromHeader = self.dictTables[tableName]["fieldFromHeader"]
                if len(fieldFromHeader) != 0:
                    for field in fieldFromHeader:
                        sql = sql + "[%s] VARCHAR(20)\n\t\t\t\t\t,"%field
                sql = sql[:-1]  # 去掉最后一个逗号
                #创建新表
                #standardizedFields = self._get_standardized_field(self.dictTables[tableName]['fieldName'],tableName)
                standardizedFields = self.dictTables[tableName]['fieldName']
                duplicatedFields = self._get_duplicated_field(standardizedFields)
                for fieldName in duplicatedFields:
                    if fieldName is not NaN:
                        sql = sql + "\n\t\t\t\t\t,[%s]  NUMERIC"%fieldName
                sql = sql + '\n\t\t\t\t\t)'
                try:
                    conn.execute(sql)
                    conn.commit()
                    print('创建数据库表%s成功' % (tableName))
                except Exception as e:
                    # 回滚
                    conn.rollback()
                    print(e,' 创建数据库表%s失败' % tableName)

                #创建索引
                sql = "CREATE INDEX IF NOT EXISTS [%s索引] on [%s] (\n\t\t\t\t\t"%(tableName,tableName)
                sql = sql + ", ".join(str(field) for field,value in self.commonFileds.items()
                                     if value.find('NOT NULL') >= 0)
                sql = sql + '\n\t\t\t\t\t)'
                try:
                    conn.execute(sql)
                    conn.commit()
                    print('创建数据库%s索引成功' % (tableName))
                except Exception as e:
                    # 回滚
                    conn.rollback()
                    print(e,' 创建数据库%s索引失败' % tableName)
        cursor.close()
        conn.close()


    def initialize(self):
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        self.clear_logging_directory(self.logging_directory)


def create_object(gConfig):
    parser = DocParserSql(gConfig)
    parser.initialize()
    return parser