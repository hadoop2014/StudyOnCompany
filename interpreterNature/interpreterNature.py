#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 6/9/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterCrawl.py
# @Note    : 用接近自然语言的解释器处理各类事务,用于处理财务数据爬取,财务数据提取,财务数据分析.
from collections import Counter
from ply import lex,yacc
import copy
import itertools
from interpreterNature.interpreterBaseClass import *


class InterpreterNature(InterpreterBase):
    def __init__(self,gConfig,interpreterDict):
        super(InterpreterNature, self).__init__(gConfig)
        self.interpreterAccounting = interpreterDict['accounting']
        self.interpreterAnalysize = interpreterDict['analysize']
        self.interpreterCrawl = interpreterDict['crawl']
        self.interpretDefine()


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
            r'[\u4E00-\u9FA5|A-Z]+'
            typeList = [key for key in local_name.keys() if key.startswith('t_')
                        and not key.startswith('t_multiLineComment')
                        and key not in ['t_VALUE','t_ignore','t_ignore_COMMENT','t_newline','t_error']]
            t.type = self._get_token_type(local_name, t.value,typeList,defaultType='VALUE')
            return t


        def t_newline(t):
            r'\n+'
            t.lexer.lineno += t.value.count("\n")


        def t_error(t):
            self.logger.info("Illegal character '%s'" % t.value[0])
            t.lexer.skip(1)

        # Build the lexer
        self.lexer = lex.lex(outputdir=self.working_directory,reflags=int(re.MULTILINE))

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


        def p_expression_create_table(p):
            '''expression : CREATE TABLE'''
            command = ' '.join([slice.value for slice in p.slice if slice.value is not None])
            self.logger.info(command)
            tableName = p[2]
            assert tableName in self.tableNames, 'tableName(%s) is invalid, which must be in %s!'%(tableName, self.tableNames)
            self._process_create_table(command)


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


        def p_expression_train_model(p):
            '''expression : TRAIN MODEL'''
            command = ' '.join([slice.value for slice in p.slice[1:] if slice.value is not None])
            self.logger.info(command)
            model = p[2]
            assert model in self.models, "model(%s) is invalid, which must be in %s" % (model, self.models)
            self._process_train_model(command)


        def p_expression_config(p):
            '''expression : CONFIG '{' configuration '}' '''
            #把names_global进行更新
            self.names_global.update(self._get_merged_names_global(self.names_global))
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
        self.parser = yacc.yacc(outputdir=self.working_directory)


    def doWork(self,lexer=None,debug=False,tracking=False):
        text = self._get_main_program()
        self.parser.parse(text,lexer=self.lexer,debug=debug,tracking=tracking)


    def _get_main_program(self):
        return self._get_text()


    def _process_parse(self,scale,isForced = False):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_full_parse!')
            return
        # 把行业分类中的公司和公司简称中的公司进行合并
        self.gConfig.update(self.names_global)
        taskResults = list()
        sourcefiles = self._get_needed_files(scale,isForced)
        sourcefiles = self._remove_black_lists(sourcefiles)
        sourcefiles = list(sourcefiles)
        sourcefiles.sort()
        for sourcefile in sourcefiles:
            self.logger.info('start process %s' % sourcefile)
            dictParameter = dict({'sourcefile': sourcefile})
            dictParameter.update(self.names_global)
            taskResult = self._process_single_parse(dictParameter)
            taskResults.append(str(taskResult))
        self.logger.info('运行结果汇总如下:\n\t\t\t\t'+'\n\t\t\t\t'.join(taskResults))


    def _process_single_parse(self,dictParameter):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_single_parse!')
            return
        self.interpreterAccounting.initialize(dictParameter)
        taskResult = self.interpreterAccounting.doWork(debug=False, tracking=False)
        return taskResult


    def _process_create_table(self,command):
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


    def _process_train_model(self, command):
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
                    sourcefiles = sourcefiles + os.listdir(source_directory)
            sourcefilesValid = [sourcefile for sourcefile in sourcefiles if self._is_file_name_valid(sourcefile)]
            sourcefilesInvalid = set(sourcefiles).difference(set(sourcefilesValid))
            if len(sourcefilesInvalid) > 0:
                for sourcefile in sourcefilesInvalid:
                     self.logger.warning('These file is can not be parse:%s'%sourcefile)
            if scale == '批量':
                sourcefilesValid = [sourcefile  for sourcefile in sourcefilesValid if self._is_file_selected(sourcefile)]
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


    def _remove_black_lists(self,sourcefiles):
        # 根据interpreterBase.json中的black_lists的配置, 将落在blaclist中的文件移除,不进行解析操作
        assert isinstance(sourcefiles,list),"sourcefile(%s) must be a list, not %s!" % (sourcefiles, type(sourcefiles))
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


    def _remove_duplicate_files(self,sourcefiles):
        #上峰水泥：2015年年度报告（更新后）.PDF和（000672）上峰水泥：2015年年度报告（更新后）.PDF并存时,则去掉前者(即去掉长度短的)
        assert isinstance(sourcefiles,list),"Parameter sourcefiles must be list!"
        nameStandardize = self.gJsonInterpreter['nameStandardize']
        dictDuplicate = dict()
        for sourcefile in sourcefiles:
            standardizedName = self._standardize(nameStandardize,sourcefile)
            if standardizedName is NaN:
                self.logger.warning('Filename %s is invalid!'%sourcefile)
                continue
            if len(dictDuplicate) == 0:
                dictDuplicate.update({standardizedName:sourcefile})
            else:
                if standardizedName in dictDuplicate.keys():
                    if len(dictDuplicate[standardizedName]) < len(sourcefile):
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
                assert isCheckOk, "行业(%s) is invalid,it must in 行业分类 %s" % (category, categorys)
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
        self.names_global['行业分类'] = {}
        self.names_local['timelist'] = NULLSTR
        self.names_local['valuelist'] = NULLSTR


def create_object(gConfig, interpreterDict):
    interpreter = InterpreterNature(gConfig, interpreterDict)
    interpreter.initialize()
    return interpreter