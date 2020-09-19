#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 6/9/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterCrawl.py
# @Note    : 用接近自然语言的解释器处理各类事务,用于处理财务数据爬取,财务数据提取,财务数据分析.
from ply import lex,yacc
import time
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


        #t_ignore = " \t\n"
        t_ignore = self.ignores
        t_ignore_COMMENT = r'#.*'


        def t_VALUE(t):
            r'[\u4E00-\u9FA5]+'
            t.type = self._get_token_type(local_name,t.value,'VALUE')
            return t


        def t_newline(t):
            r'\n+'
            t.lexer.lineno += t.value.count("\n")


        def t_error(t):
            self.logger.info("Illegal character '%s'" % t.value[0])
            t.lexer.skip(1)

        # Build the lexer
        self.lexer = lex.lex(outputdir=self.working_directory,reflags=int(re.MULTILINE))

        # dictionary of names
        self.names = {}


        def p_statement_expression(p):
            '''statement : statement expression
                         | expression'''
            pass


        def p_expression_batch_parse(p):
            '''expression : SCALE EXECUTE PARSE'''
            if p[1] == "全量":
                self._process_full_parse()
            elif p[1] == "批量":
                self._process_batch_parse()
            elif p[1] == "单次":
                dictParmeter = dict({'sourcefile':self.gConfig['sourcefile']})
                self._process_single_parse(dictParmeter)
            else:
                self.logger.info("Mistakes in grammar,the SCALE (%s) is not a valid token in [全量,批量,单次]"%p[1])



        def p_expression_create_table(p):
            '''expression : CREATE TABLE'''
            command = ' '.join([slice.value for slice in p.slice if slice.value is not None])
            self._process_create_table(command)


        def p_expression_visualize(p):
            '''expression : SCALE VISUALIZE TABLE'''
            command = ' '.join([slice.value for slice in p.slice[1:] if slice.value is not None])
            self._process_visualize_table(command)

        def p_expression_crawl(p):
            '''expression : SCALE CRAWL WEBSITE'''
            command = ' '.join([slice.value for slice in p.slice[1:] if slice.value is not None])
            self._process_crawl_finance(command)


        def p_expression_config(p):
            '''expression : CONFIG '{' configuration '}' '''
            p[0] = p[1] +'{ ' + p[3] +' }'


        def p_configuration(p):
            '''configuration : configuration  configuration '''
            p[0] = p[1] + ';' + p[2]


        def p_configuration_value(p):
            '''configuration : PARAMETER ':' NUMERIC
                             | PARAMETER ':' time
                             | PARAMETER ':' value'''
            if p.slice[3].type == 'NUMERIC':
                self.names.update({p[1]:list([p[3]])})
            elif p.slice[3].type == 'time':
                self.names.update({p[1]:self.names['timelist']})
            elif p.slice[3].type == 'value':
                self.names.update({p[1]:self.names['valuelist']})
            self.logger.info("fetch config %s : %s"%(p[1],p[3]))
            p[0] = p[1] + ':' + p[3]


        def p_time(p):
            '''time : TIME
                    | TIME '-' TIME '''
            if len(p.slice) == 4:
                assert self._is_matched('\\d+年',p[1]) and self._is_matched('\\d+年',p[3])\
                    ,"parameter %s or %s is not invalid TIME"%(p[1],p[3])
                timelist = [str(year) + '年'  for year in range(int(p[1].split('年')[0]),int(p[3].split('年')[0]) + 1)]
                p[0] = p[1] + p[2] + p[3]
            else:
                assert self._is_matched('\\d+年', p[1]) , "parameter %s is not invalid TIME" % (p[1])
                timelist = list([p[1]])
                p[0] = p[1]
            self.names.update({'timelist':timelist})

        def p_value(p):
            '''value :  value ',' VALUE
                     | VALUE'''
            if len(p.slice) == 4:
                valuelist = self.names['valuelist'] + list([p[3]])
                p[0] = p[1] + p[2] + p[3]
            else:
                valuelist = list([p[1]])
                p[0] = p[1]
            self.names.update({'valuelist':valuelist})

        def p_error(p):
            if p:
                print("Syntax error at '%s:%s'" % (p.value,p.type))
                self.logger.error("Syntax error at '%s:%s'" % (p.value, p.type))
            else:
                print("Syntax error at EOF page")
                self.logger.error("Syntax error at EOF page")


        # Build the docparser
        self.parser = yacc.yacc(outputdir=self.working_directory)


    def _get_token_type(self, local_name,value, defaultType='VALUE'):
        #解决保留字和VALUE的冲突问题
        type = defaultType
        for key,content in local_name.items():
            if key.startswith('t_') and key not in ['t_'+defaultType,'t_ignore','t_ignore_COMMENT','t_newline','t_error']:
                match = re.search(local_name[key],value)
                if match is not None:
                    type = key.split('_')[-1]
                    break
        return type


    def doWork(self,lexer=None,debug=False,tracking=False):
        text = self._get_main_program()
        self.parser.parse(text,lexer=self.lexer,debug=debug,tracking=tracking)


    def _get_main_program(self):
        return self._get_text()


    def _process_full_parse(self):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_full_parse!')
            return
        taskResults = list()
        source_directory = os.path.join(self.gConfig['data_directory'], self.gConfig['source_directory'])
        sourcefiles = os.listdir(source_directory)
        for sourcefile in sourcefiles:
            self.logger.info('start process %s' % sourcefile)
            #self.gConfig.update({'sourcefile': sourcefile})
            if not self._is_file_name_valid(sourcefile):
                self.logger.warning("%s is not a valid file" % sourcefile)
                continue
            dictParameter = dict({'sourcefile': sourcefile})
            taskResult = self._process_single_parse(dictParameter)
            taskResults.append(taskResult)
        self.logger.info(taskResults)


    def _process_batch_parse(self):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_batch_parse!')
            return
        taskResults = list()
        source_directory = os.path.join(self.gConfig['data_directory'], self.gConfig['source_directory'])
        sourcefiles = os.listdir(source_directory)
        for sourcefile in sourcefiles:
            if not self._is_file_name_valid(sourcefile):
                self.logger.warning("%s is not a valid file" % sourcefile)
                continue
            if self._is_file_selcted(sourcefile):
                self.logger.info('start process %s' % sourcefile)
                #self.gConfig.update({'sourcefile': sourcefile})
                dictParameter = dict({'sourcefile': sourcefile})
                taskResult = self._process_single_parse(dictParameter)
                taskResults.append(taskResult)
        self.logger.info(taskResults)


    def _process_single_parse(self,dictParameter):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_single_parse!')
            return
        self.interpreterAccounting.initialize(dictParameter)
        taskResult = self.interpreterAccounting.doWork(debug=False, tracking=False)
        return taskResult


    def _is_file_selcted(self,sourcefile):
        assert self.names['公司简称'] != NULLSTR and self.names['报告类型'] != NULLSTR and self.names['报告时间'] != NULLSTR\
            ,"parameter 公司简称,报告类型,报告年度 must not be NULL in 批量处理程序"

        isFileSelected = self._is_matched('|'.join(self.names['公司简称']),sourcefile) \
                         and self._is_matched('|'.join(self.names['报告类型']),sourcefile) \
                         and self._is_matched('|'.join(self.names['报告时间']),sourcefile)
        return isFileSelected


    def _process_create_table(self,command):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_single_analysize!')
            return
        pass
        self.interpreterAnalysize.doWork(command)


    def _process_visualize_table(self,command):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_single_analysize!')
            return
        pass
        self.gConfig.update(self.names)
        self.interpreterAnalysize.initialize(self.gConfig)
        self.interpreterAnalysize.doWork(command)

    def _process_crawl_finance(self,command):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_single_analysize!')
            return
        pass
        self.gConfig.update(self.names)
        self.interpreterCrawl.initialize(self.gConfig)
        self.interpreterCrawl.doWork(command)


    def initialize(self):
        self.names['公司简称'] = NULLSTR
        self.names['报告时间'] = NULLSTR
        self.names['报告类型'] = NULLSTR
        self.names['timelist'] = NULLSTR
        self.names['valuelist'] = NULLSTR


def create_object(gConfig, interpreterDict):
    interpreter = InterpreterNature(gConfig, interpreterDict)
    interpreter.initialize()
    return interpreter