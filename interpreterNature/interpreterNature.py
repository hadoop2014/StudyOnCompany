#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 6/9/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterCrawl.py
# @Note    : 用接近自然语言的解释器处理各类事务,用于处理财务数据爬取,财务数据提取,财务数据分析.
import multiprocessing
from collections import Counter
from ply import lex,yacc
import copy
import itertools
import time
from interpreterNature.interpreterBaseClass import *

class Sqlite(SqilteBase):

    def _get_company_code_list(self,conn,tableName):
        companyCodeList = None
        sql = "select distinct 公司代码 from {}".format(tableName)
        try:
            result = conn.execute(sql).fetchall()
            if len(result) > 0 and len(result[0]) > 0:
                #companyCodeList = result[0]
                companyCodeList = [code[0] for code in result]
        except Exception as e:
            print(e)
            self.logger.error('failed to get max & min trading data from sql:%s' % sql)
        return companyCodeList

    def _write_to_sqlite3(self,dataFrame:DataFrame,commonFields, tableName):
        conn = self._get_connect()
        sql_df = dataFrame
        companyCodeList = self._get_company_code_list(conn, tableName)
        if companyCodeList is not None:
            companyCodeNew = sql_df['公司代码'].values.tolist()
            companyCodeDiff = set(companyCodeNew).difference(set(companyCodeList))
            if len(companyCodeDiff) > 0:
                #sql_df = sql_df[sql_df['公司代码'] in companyCodeDiff]
                sql_df = sql_df[sql_df['公司代码'].isin(companyCodeDiff)]
                if not sql_df.empty:
                    sql_df.to_sql(name=tableName, con=conn, if_exists='append', index=False)
                    conn.commit()
                    self.logger.info("insert into {} at {}!".format(tableName, utile.get_time_now()))
        else:
            sql_df.to_sql(name=tableName, con=conn, if_exists='replace', index=False)
            conn.commit()
            self.logger.info("insert into {} at {}!".format(tableName, utile.get_time_now()))
        conn.close()


class InterpreterNature(InterpreterBase):
    def __init__(self,gConfig,interpreterDict):
        super(InterpreterNature, self).__init__(gConfig)
        self.interpreterAccounting = interpreterDict['accounting']
        self.interpreterAnalysize = interpreterDict['analysize']
        self.interpreterCrawl = interpreterDict['crawl']
        self.interpretDefine()
        self.database = self.create_database(Sqlite)


    def interpretDefine(self):
        tokens = self.tokens
        literals = self.literals
        # Tokens
        #采用动态变量名
        local_name = locals()
        for token in self.tokens:
            local_name['t_'+token] = self.dictTokens[token]
        self.logger.info('\n'+str({key:value for key,value in local_name.items() if key.split('_')[-1] in tokens}).replace("',","'\n"))

        # 如下代码实现多行注释
        states = (
            ('multiLineComment', 'exclusive'),
        )

        t_multiLineComment_ignore =  r':'

        def t_multiLineComment(t):
            r'/\*'
            t.lexer.begin('multiLineComment')

        def t_multiLineComment_end(t):
            r'\*/'
            t.lexer.begin('INITIAL')

        def t_multiLineComment_newline(t):
            r'\n'
            pass

        # catch (and ignore) anything that isn't end-of-comment
        def t_multiLineComment_content(t):
            r'[^/\*]+'
            pass

        def t_multiLineComment_error(t):
            self.logger.info("Illegal character '%s'" % t.value[0])
            t.lexer.skip(1)
        # 多行注释代码结束

        t_ignore = self.ignores
        t_ignore_COMMENT = r'#.*'


        def t_VALUE(t):
            r'[\u4E00-\u9FA5|A-Z|0-9]+'
            typeList = [key for key in local_name.keys() if key.startswith('t_')
                        and not key.startswith('t_multiLineComment')
                        and key not in ['t_VALUE','t_ignore','t_ignore_COMMENT','t_newline','t_error','t_NUMERIC']]
            t.type = self._get_token_type(local_name, t.value,typeList,defaultType='VALUE')
            return t


        def t_newline(t):
            r'\n+'
            t.lexer.lineno += t.value.count("\n")


        def t_error(t):
            self.logger.info("Illegal character '%s'" % t.value[0])
            t.lexer.skip(1)

        # Build the lexer
        self.lexer = lex.lex(outputdir=self.workingspace.directory,reflags=int(re.MULTILINE))

        # dictionary of names_global
        self.names_global = {}
        self.names_local = {}


        def p_statement_expression(p):
            '''statement : statement expression
                         | expression'''
            pass


        def p_expression_batch_parse(p):
            '''expression : SCALE EXECUTE PARSE'''
            command = ' '.join([slice.value for slice in p.slice if slice.value is not None])
            self.logger.info(command)
            scale = p[1]
            isForced = (p[2] == '强制运行')
            self._process_parse(scale,isForced)


        def p_expression_batch_analysize(p):
            '''expression : SCALE EXECUTE ANALYSIZE'''
            command = ' '.join([slice.value for slice in p.slice if slice.value is not None])
            self.logger.info(command)
            self._process_analysize(command)


        def p_expression_manipulate_table(p):
            '''expression : MANIPULATE TABLE'''
            command = ' '.join([slice.value for slice in p.slice if slice.value is not None])
            self.logger.info(command)
            tableName = p[2]
            assert tableName in self.tableNames, 'tableName(%s) is invalid, which must be in %s!'%(tableName, self.tableNames)
            self._process_manipulate_table(command)


        def p_expression_visualize(p):
            '''expression : SCALE VISUALIZE TABLE'''
            command = ' '.join([slice.value for slice in p.slice[1:] if slice.value is not None])
            self.logger.info(command)
            tableName = p[3]
            assert tableName in self.tableNames, 'tableName(%s) is invalid, which must be in %s!' % (
            tableName, self.tableNames)
            self._process_visualize_table(command)


        def p_expression_crawl(p):
            '''expression : SCALE CRAWL WEBSITE'''
            command = ' '.join([slice.value for slice in p.slice[1:] if slice.value is not None])
            self.logger.info(command)
            website = p[3]
            assert website in self.websites,"website(%s) is invalid, which must be in %s" % (website, self.websites)
            self._process_crawl_finance(command)


        def p_expression_import(p):
            '''expression : SCALE IMPORT TABLE'''
            command = ' '.join([slice.value for slice in p.slice[1:] if slice.value is not None])
            self.logger.info(command)
            tableName = p[3]
            assert tableName in self.tableNames,"table(%s) is invalid, which must be in %s" % (tableName, self.tableNames)
            self._process_import_finance(command)


        def p_expression_handle_model(p):
            '''expression : HANDLE MODEL'''
            command = ' '.join([slice.value for slice in p.slice[1:] if slice.value is not None])
            self.logger.info(command)
            model = p[2]
            assert model in self.models, "model(%s) is invalid, which must be in %s" % (model, self.models)
            self._process_handle_model(command)


        def p_expression_config(p):
            '''expression : CONFIG '{' configuration '}' '''
            #把names_global进行更新
            dictMergedNames = self._get_merged_names_global(self.names_global)
            self.names_global.update(dictMergedNames)
            self._process_industry_category(dictMergedNames['行业分类'],tableName='行业分类数据')
            p[0] = p[1] +'{ ' + p[3] +' }'


        def p_configuration(p):
            '''configuration : configuration  configuration '''
            p[0] = p[1] + ';' + p[2]


        def p_configuration_value(p):
            '''configuration : PARAMETER ':' NUMERIC
                             | PARAMETER ':' time
                             | PARAMETER ':' value
                             | CATEGORY ':' value'''
            if p.slice[1].type == 'PARAMETER':
                if p.slice[3].type == 'NUMERIC':
                    self.names_global.update({p[1]:list([p[3]])})
                elif p.slice[3].type == 'time':
                    self.names_global.update({p[1]:self.names_local['timelist']})
                elif p.slice[3].type == 'value':
                    if isinstance(self.names_global[p[1]],list):
                        self.names_global.update({p[1]:self.names_global[p[1]] + self.names_local['valuelist']})
                    else:
                        self.names_global.update({p[1]:self.names_local['valuelist']})
                self._parameter_check(p[1], self.names_global[p[1]])
            elif p.slice[1].type == 'CATEGORY':
                key = self.gJsonInterpreter['CATEGORY']
                self.names_global[key].update({p[1]:self.names_local['valuelist']})
                self._parameter_check(key, self.names_global[key][p[1]])
            self.logger.info("fetch config %s : %s"%(p[1],p[3]))
            p[0] = p[1] + ':' + p[3]


        def p_time(p):
            '''time : TIME
                    | TIME '-' TIME '''
            if len(p.slice) == 4:
                timelist = [str(year) + '年'  for year in range(int(p[1].split('年')[0]) - 1,int(p[3].split('年')[0]) + 1)]
                p[0] = p[1] + p[2] + p[3]
            else:
                timelist = list([p[1]])
                p[0] = p[1]
            self.names_local.update({'timelist':timelist})


        def p_value(p):
            '''value : value ',' VALUE
                     | VALUE'''
            if len(p.slice) == 4:
                valuelist = self.names_local['valuelist'] + list([p[3]])
                p[0] = p[1] + p[2] + p[3]
            else:
                valuelist = list([p[1]])
                p[0] = p[1]
            self.names_local.update({'valuelist':valuelist})


        def p_error(p):
            if p:
                print("Syntax error at '%s:%s'" % (p.value,p.type))
                self.logger.error("Syntax error at '%s:%s'" % (p.value, p.type))
            else:
                print("Syntax error at EOF page")
                self.logger.error("Syntax error at EOF page")


        # Build the docparser
        self.parser = yacc.yacc(outputdir=self.workingspace.directory)


    def doWork(self,lexer=None,debug=False,tracking=False):
        text = self._get_main_program()
        self.parser.parse(text,lexer=self.lexer,debug=debug,tracking=tracking)


    def _get_main_program(self):
        return self._get_text()


    def _process_industry_category(self,dictIndustryCategory, tableName='行业分类数据'):
        """
            args:
                dictIndustryCategory - 行业分类数据:
                '''
                行业分类: 公司名称
                '''
                tableName - 待写入数据库的表名,表结构在interpreterBase.json中定义:
                '''
                默认表名为 行业分类数据,
                '''
            reutrn:
                NULL - 处理规则:
                '''
                1) dictIndustryCategory 处理为: 报告时间, 公司代码,公司简称, 行业分类.
                2) 将处理后的数据写入sqlite3由tableName指定的表中,默认写入 "行业分类数据"
                '''
        """
        companys = list(set(dictIndustryCategory.keys()))
        companyCodes = self._get_stock_list(companys)
        dataFrameCompanyCodes = pd.DataFrame(companyCodes, columns=self.gJsonBase['stockcodeHeader'])
        # 增加一列 上报时间,取当前日期
        dataFrameCompanyCodes['报告时间'] = utile.get_time_now()
        dataFrameIndustryCategery = pd.DataFrame([[key,value] for key,value in dictIndustryCategory.items()], columns=['公司简称','行业分类'])
        dataFrameMerged = pd.merge(dataFrameCompanyCodes,dataFrameIndustryCategery,how='left',on=['公司简称'])
        columnsName = self._get_merged_columns(tableName)
        dataFrameMerged = dataFrameMerged[columnsName]
        self.database._write_to_sqlite3(dataFrameMerged,self.commonFields, tableName)

    '''
    def _write_to_sqlite3(self,dataFrame:DataFrame,commonFields, tableName):
        conn = self.database._get_connect()
        sql_df = dataFrame
        companyCodeList = self._get_company_code_list(conn, tableName)
        if companyCodeList is not None:
            companyCodeNew = sql_df['公司代码'].values.tolist()
            companyCodeDiff = set(companyCodeNew).difference(set(companyCodeList))
            if len(companyCodeDiff) > 0:
                #sql_df = sql_df[sql_df['公司代码'] in companyCodeDiff]
                sql_df = sql_df[sql_df['公司代码'].isin(companyCodeDiff)]
                if not sql_df.empty:
                    sql_df.to_sql(name=tableName, con=conn, if_exists='append', index=False)
                    conn.commit()
                    self.logger.info("insert into {} at {}!".format(tableName, self._get_time_now()))
        else:
            sql_df.to_sql(name=tableName, con=conn, if_exists='replace', index=False)
            conn.commit()
            self.logger.info("insert into {} at {}!".format(tableName, self._get_time_now()))
        conn.close()
    '''
    '''
    def _get_company_code_list(self,conn,tableName):
        companyCodeList = None
        sql = "select distinct 公司代码 from {}".format(tableName)
        try:
            result = conn.execute(sql).fetchall()
            if len(result) > 0 and len(result[0]) > 0:
                #companyCodeList = result[0]
                companyCodeList = [code[0] for code in result]
        except Exception as e:
            print(e)
            self.logger.error('failed to get max & min trading data from sql:%s' % sql)
        return companyCodeList
    '''

    def _process_analysize(self,command):
        """
        explain:
            用于执行股票分析/指数分析的功能, 根据 .nature文件中 参数配置{} 中配置的指数简称, 从sqlite3中读取 股票交易数据, 最终调用
            interpreterAnalysize.stockAnalysize.py进行股票交易数据的处理,处理结果写入 指数趋势分析表,... 等数据表中
        Example:
            批量 执行 指数趋势分析
            全量 执行 指数趋势分析(暂不支持)
            单次 执行 指数趋势分析(暂不支持)
        args:
            commond - 待执行语句,示例: 批量 执行 指数趋势分析
        return:
            无
        """
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_full_parse!')
            return
        # 把行业分类中的公司和公司简称中的公司进行合并
        self.gConfig.update(self.names_global)
        self.interpreterAnalysize.initialize(self.gConfig)
        self.interpreterAnalysize.doWork(command)


    def _process_parse(self,scale,isForced = False):
        """
        explain:
            用于执行财务报表解析的功能, 根据 .nature文件中 参数配置{} 中配置的公司简称, 报告类型, 报告时间, 从 data_directory中读取指定
            的财报文件, 根据interpreterBase.Json的black_lists的配置移除无需处理的文件,然后由_process_single_parse进行财报解析.
        Example:
            批量 执行 财务报表解析
            全量 执行 财务报表解析
            单次 执行 财务报表解析
            单次 强制执行 财务报表解析
        args:
            scale - 执行规模,为: 全量, 批量, 单次
            isForced - 是否强制执行的标识, True: 本次为强制执行, 会清空checkpoint中问记录, 重新执行指定的财报解析
        return:
            无
        """
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_full_parse!')
            return
        startTime = time.time()
        # 把行业分类中的公司和公司简称中的公司进行合并
        self.gConfig.update(self.names_global)
        taskResults = list()
        sourcefiles = self._get_needed_files(scale,isForced)
        sourcefiles = self._remove_black_lists(scale,list(sourcefiles))
        sourcefiles = list(sourcefiles)
        sourcefiles.sort()
        for sourcefile in sourcefiles:
            self.logger.info('start process %s' % sourcefile)
            dictParameter = dict({'sourcefile': sourcefile})
            dictParameter.update(self.names_global)
            '''
            if self.multiprocessingIsOn:
                # 这里采用多进程编程,充分使用多核心并行
                # 采用信号量控制同时运行的进程数量,默认等于CPU数量
                self.semaphore.acquire()
                processParse = multiprocessing.Process(target=self._process_single_parse,args=(dictParameter,))
                #taskResult = processPool.apply_async(self._process_single_parse,args=(self,dictParameter)) # Pool不舍用类方法
                processList.append(processParse)
                processParse.start()
            else:
                taskResult = self._process_single_parse(dictParameter)
                taskResults.append(str(taskResult))
            '''
            # 此处对_process_single_parse采用了多进程
            taskResult = self._process_single_parse(dictParameter)
            taskResults.append(str(taskResult))
        # 当采用多进程编程时,此处需要关闭多进程, 同时获取multiprocessing.Queue()中的函数返回值
        taskResults = Multiprocess.release()
        self.logger.info('运行结果汇总如下(总耗时%.4f秒):\n\t\t\t\t'%(time.time()-startTime) + '\n\t\t\t\t'.join(taskResults))


    @Multiprocess
    def _process_single_parse(self,dictParameter):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_single_parse!')
            return
        self.interpreterAccounting.initialize(dictParameter)
        taskResult = self.interpreterAccounting.doWork(debug=False, tracking=False)
        return taskResult


    def _process_manipulate_table(self, command):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_single_analysize!')
            return
        assert self.names_global['报告时间'] != NULLSTR and self.names_global['报告类型'] != NULLSTR \
            , "报告时间,报告类型为空,必须在参数配置中明确配置!"
        self.gConfig.update(self.names_global)
        self.interpreterAnalysize.initialize(self.gConfig)
        self.interpreterAnalysize.doWork(command)


    def _process_visualize_table(self,command):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_single_analysize!')
            return
        assert self.names_global['报告时间'] != NULLSTR and self.names_global['报告类型'] != NULLSTR\
            ,"报告时间,报告类型为空,必须在参数配置中明确配置!"
        self.gConfig.update(self.names_global)
        self.interpreterAnalysize.initialize(self.gConfig)
        self.interpreterAnalysize.doWork(command)


    def _process_handle_model(self, command):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_single_analysize!')
            return
        self.gConfig.update(self.names_global)
        self.interpreterAnalysize.initialize(self.gConfig)
        self.interpreterAnalysize.doWork(command)


    def _process_crawl_finance(self,command):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_single_analysize!')
            return
        assert self.names_global['报告时间'] != NULLSTR and self.names_global['报告类型'] != NULLSTR\
            ,"报告时间,报告类型为空,必须在参数配置中明确配置!"
        self.gConfig.update(self.names_global)
        # 爬取时在时间上少设置1年,因为2020年的时间本来就会爬取2019年的数据
        if len(self.gConfig['报告时间']) > 1:
            self.gConfig.update({'报告时间': self.names_global['报告时间'][1:]})
        self.interpreterCrawl.initialize(self.gConfig)
        self.interpreterCrawl.doWork(command)


    def _process_import_finance(self,command):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_single_analysize!')
            return
        assert self.names_global['报告时间'] != NULLSTR and self.names_global['报告类型'] != NULLSTR\
            ,"报告时间,报告类型为空,必须在参数配置中明确配置!"
        self.gConfig.update(self.names_global)
        # 爬取时在时间上少设置1年,因为2020年的时间本来就会爬取2019年的数据
        self.gConfig.update({'报告时间': self.names_global['报告时间'][1:]})
        self.interpreterCrawl.initialize(self.gConfig)
        self.interpreterCrawl.doWork(command)


    def _get_merged_names_global(self,names_global):
        # 把行业分类中的公司简称全部合并
        gConfig = copy.deepcopy(names_global)
        for categry,company in gConfig['行业分类'].items():
            gConfig.setdefault('公司简称', []).extend(company)
        if len(gConfig['公司简称']) != len(set(gConfig['公司简称'])):
            dictDuplicate = dict(Counter(gConfig['公司简称']))
            self.logger.warning('以下公司名称在参数配置中重复了:%s'%([key for key,value in dictDuplicate.items() if value > 1]))
        #把行业分类的dict进行转置
        dictCategory = dict(itertools.chain.from_iterable([list(zip(valueList,[key]*len(valueList))) for key,valueList in gConfig['行业分类'].items()]))
        gConfig.update({"行业分类":dictCategory})
        return gConfig


    def _get_needed_files(self,scale,isForced = False):
        """
        args:
            scale - 语句中指示的处理规模:
                '''
                单次,批量,全量
                '''
            isForced - 语句中指示的处理方式:
                '''
                True - 强制执行, 清除checkpoint中指定的sourcefiles,后续进行这些报表的解析
                False - 如果sourcefiles已经在checkpoint中已经有记录,则后续不再进行这些报表解析
                '''
        reutrn:
            sourcefiles - 需要下一步进行财报解析的文件列表,处理规则:
                '''
                1) 从data_directory中读取所有文件,剔除无效文件;
                2) _remove_exclude_files移除在interpreterBase.json:'black_lists':'例外文件'中配置的文件;
                3) _remove_duplicate_files移除重复文件, 如果'昊海生科：2019年年度报告','昊海生科：2019年年度报告（修订版）',保留后者;
                4) 如果isForced=False,则移除已经在checkpoint记录中的文件,这些文件不再继续解析;
                '''
        """
        sourcefiles = list()
        if scale == '单次':
            sourcefilesValid = list([self.gConfig['sourcefile']])
        else:
            if self.names_global['报告类型'] == NULLSTR:
                source_directory = os.path.join(self.gConfig['data_directory'], self.gConfig['source_directory'])
                sourcefiles = os.listdir(source_directory)
            else:
                for type  in self.names_global['报告类型']:
                    source_directory = self._get_path_by_report_type(type)
                    if source_directory != NULLSTR:
                        sourcefiles = sourcefiles + os.listdir(source_directory)
            sourcefilesValid = [sourcefile for sourcefile in sourcefiles if self._is_file_name_valid(sourcefile)]
            sourcefilesInvalid = set(sourcefiles).difference(set(sourcefilesValid))
            if len(sourcefilesInvalid) > 0:
                for sourcefile in sourcefilesInvalid:
                     self.logger.warning('These file is can not be parse:%s'%sourcefile)
            if scale == '批量':
                sourcefilesValid = [sourcefile  for sourcefile in sourcefilesValid if self._is_file_selected(sourcefile)]
            sourcefilesValid = self._remove_exclude_files(sourcefilesValid)
            sourcefilesValid = self._remove_duplicate_files(sourcefilesValid)
        if isForced == False:
            checkpoint = self.interpreterAccounting.docParser.get_checkpoint()
            if isinstance(checkpoint,list) and len(checkpoint) > 0:
                sourcefilesRemainder = set(sourcefilesValid).difference(set(checkpoint))
                sourcefilesDone = set(sourcefilesValid).difference(set(sourcefilesRemainder))
                if len(sourcefilesDone) > 0:
                    for sourcefile in sourcefilesDone:
                        self.logger.info('the file %s is already in checkpointfile,no need to process!'%sourcefile)
                sourcefiles = sourcefilesRemainder
            else:
                sourcefiles = sourcefilesValid
        else:
            self.logger.info('force to start process........\n')
            sourcefiles = list(sourcefilesValid)
            self.interpreterAccounting.docParser.remove_checkpoint_files(sourcefiles)
        return sourcefiles


    def _remove_black_lists(self,scale,sourcefiles):
        # 根据interpreterBase.json中的black_lists的配置, 将落在blaclist中的文件移除,不进行解析操作
        assert isinstance(sourcefiles,list),"sourcefile(%s) must be a list, not %s!" % (sourcefiles, type(sourcefiles))
        if scale == '单次':
            #如果是单次执行,则不做处理,直接返回
            return sourcefiles
        resultSourcefiles = [sourcefile for sourcefile in sourcefiles if not self._is_file_in_black_lists(sourcefile)]
        diffSourcefiles = set(sourcefiles).difference(set(resultSourcefiles))
        if len(diffSourcefiles) > 0:
            self.logger.info('these file is in black_lists of interpreterBase.json, now no need to process:\n\t%s'
                             % '\n\t'.join(sorted(diffSourcefiles)))
        return resultSourcefiles


    def _is_file_in_black_lists(self,sourcefile):
        patterns = list()
        for company,value in self.gJsonBase['black_lists'].items():
            if isinstance(value,dict):
                for type,times in value.items():
                    patterns = patterns + [company + '：' + time + type for time in times]
        matchPattern = '|'.join(patterns)
        isFileMatched = self._is_matched(matchPattern, sourcefile)
        return isFileMatched


    def _remove_exclude_files(self,sourcefiles):
        """
        args:
            sourcefiles - 带进行财务报表解析的文件列表:
        reutrn:
            sourcefiles - 将符合在interpreterBase.json:'black_lists':'例外文件' 和 '例外文件特征'定义的文件移除:
            '''
            1) interpreterBase.json:'black_lists':'例外文件',这些文件是错误的,可以用其他文件取代;
            2) interpreterBase.json:'black_lists':'例外文件特征',具有这些特征的文件是不用解析的,可以用其他文件取代,示例:
                （603707）健友股份：2018年年度报告（已取消）.PDF)
            '''
        """
        assert isinstance(sourcefiles,list),"Parameter sourcefiles must be list!"
        excludeFiles = self.gJsonBase['black_lists']['例外文件']
        rawSourceFiles = sourcefiles
        if len(excludeFiles) > 0:
            sourcefiles = [sourcefile for sourcefile in sourcefiles if sourcefile not in excludeFiles]
        excludeFilesPattern = self.gJsonBase['black_lists']['例外文件特征']
        excludeFilesPattern = '|'.join(excludeFilesPattern)
        if excludeFilesPattern != NULLSTR:
            sourcefiles = [sourcefile for sourcefile in sourcefiles if re.search(excludeFilesPattern,sourcefile) is None]
        removedFiles = set(rawSourceFiles).difference(set(sourcefiles))
        if len(removedFiles) > 0:
            self.logger.info('these file is in black_lists 例外文件 and 例外文件特征 of interpreterBase.json, now no need to process:\n\t%s'
                             % '\n\t'.join(sorted(removedFiles)))
        return sourcefiles


    def _remove_duplicate_files(self,sourcefiles):
        #上峰水泥：2015年年度报告（更新后）.PDF和（000672）上峰水泥：2015年年度报告（更新后）.PDF并存时,则去掉前者(即去掉长度短的)
        assert isinstance(sourcefiles,list),"Parameter sourcefiles must be list!"
        filenameStandardize = self.gJsonBase['filenameStandardize']
        dictDuplicate = dict()
        for sourcefile in sourcefiles:
            standardizedName = self._standardize(filenameStandardize,sourcefile)
            company,time, reportType,code = self._get_company_time_type_code_by_filename(standardizedName) #解决白云山:2020年第一季度报告,和白云山:2020年第一季度报告全文,取后者
            if company is not NaN and time is not NaN and reportType is not NaN:
                standardizedName = company + '：'+time + reportType
            if standardizedName is NaN:
                self.logger.warning('Filename %s is invalid!'%sourcefile)
                continue
            if len(dictDuplicate) == 0:
                dictDuplicate.update({standardizedName:sourcefile})
            else:
                if standardizedName in dictDuplicate.keys():
                    if len(dictDuplicate[standardizedName]) < len(sourcefile):
                        # 相同年份的财报,取文件名长的文件,示例: （300529）健帆生物：2018年年度报告.PDF 和 （603707）健友股份：2018年年度报告（修订版）.PDF,取后则
                        self.logger.info("File %s is duplicated and replaced by %s"
                                         %(dictDuplicate[standardizedName],sourcefile))
                        dictDuplicate.update({standardizedName:sourcefile})
                else:
                    dictDuplicate.update({standardizedName: sourcefile})
        sourcefiles = dictDuplicate.values()
        return sourcefiles


    def _is_file_selected(self, sourcefile):
        assert self.names_global['公司简称'] != NULLSTR and self.names_global['报告类型'] != NULLSTR and self.names_global['报告时间'] != NULLSTR\
            ,"parameter 公司简称,报告类型,报告年度 must not be NULL in 批量处理程序"
        isFileSelected = self._is_matched('|'.join(self.names_global['公司简称']), sourcefile) \
                         and self._is_matched('|'.join(self.names_global['报告类型']), sourcefile) \
                         and self._is_matched('|'.join(self.names_global['报告时间']), sourcefile)
        return isFileSelected


    def _parameter_check(self,key,values):
        assert isinstance(values,list),"values(%s) is not a list"%values
        isCheckOk = False
        categorys = self.gJsonInterpreter['行业分类']
        if key == '行业分类':
            for category in self.names_global[key].keys():
                isCheckOk =  category in categorys.keys()
                assert isCheckOk, "行业(%s) is invalid,it must in 行业分类 %s" % (category, categorys.keys())
        stockIndexes = self.gJsonBase['stockindex']
        if key == '指数简称':
            for stockIndex in self.names_global[key]:
                isCheckOk = stockIndex in stockIndexes.keys()
                assert isCheckOk, "指数(%s) is invalid,it must in stockindex %s" %(stockIndex, stockIndexes.keys())
        parametercheck = self.gJsonInterpreter['parametercheck']
        for value in values:
            if key in parametercheck.keys():
                #isCheckOk = self._is_matched(parametercheck[key],value)
                isCheckOk = self._standardize(parametercheck[key],value) == value
            assert isCheckOk,"Value(%s) is invalid,it must in list %s"%(value,parametercheck[key].split('|'))


    def initialize(self):
        self.names_global['公司简称'] = []
        self.names_global['报告时间'] = NULLSTR
        self.names_global['报告类型'] = NULLSTR
        self.names_global['指数简称'] = []
        self.names_global['行业分类'] = {}
        self.names_local['timelist'] = NULLSTR
        self.names_local['valuelist'] = NULLSTR


def create_object(gConfig, interpreterDict):
    interpreter = InterpreterNature(gConfig, interpreterDict)
    interpreter.initialize()
    return interpreter