#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 6/9/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterNature.py
# @Note    : 用接近自然语言的解释器处理各类事务,用于处理财务数据爬取,财务数据提取,财务数据分析.
from ply import lex,yacc
import time
from interpreterNature.interpreterBaseClass import *

class InterpreterNature(InterpreterBase):
    def __init__(self,gConfig,interpreterDict):
        super(InterpreterNature, self).__init__(gConfig)
        self.interpreterAccounting = interpreterDict['accounting']
        self.interpreterAnalysize = interpreterDict['analysize']
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

        def t_newline(t):
            r'\n+'
            t.lexer.lineno += t.value.count("\n")

        def t_error(t):
            print("Illegal character '%s'" % t.value[0])
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
            '''expression : BATCH PARSE'''
            self._process_batch_parse()

        def p_expression_single_parse(p):
            '''expression : SINGLE PARSE'''
            self._process_single_parse()

        def p_expression_batch_analysize(p):
            '''expression : BATCH ANALYSIZE'''
            self._process_batch_analysize()

        def p_expression_single_analysize(p):
            '''expression : SINGLE ANALYSIZE'''
            self._process_single_analysize()

        def p_expression_execute_analysize(p):
            '''expression : EXECUTE ANALYSIZE'''
            self._process_single_analysize()

        def p_error(p):
            if p:
                print("Syntax error at '%s:%s'" % (p.value,p.type))
            else:
                print("Syntax error at EOF page")

        # Build the docparser
        self.parser = yacc.yacc(outputdir=self.working_directory)

    def doWork(self,lexer=None,debug=False,tracking=False):
        #for data in docParser:
        #    self.currentPageNumber = docParser.index
        #    text = docParser._get_text(data)
        text = self._get_main_program()
        self.parser.parse(text,lexer=self.lexer,debug=debug,tracking=tracking)

    def _get_main_program(self):
        return self._get_text()

    def _process_batch_parse(self):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_batch_parse!')
            return
        taskResults = list()
        source_directory = os.path.join(self.gConfig['data_directory'], self.gConfig['source_directory'])
        sourcefiles = os.listdir(source_directory)
        for sourcefile in sourcefiles:
            self.logger.info('start process %s' % sourcefile)
            self.gConfig.update({'sourcefile': sourcefile})
            if not self._is_file_name_valid(sourcefile):
                self.logger.warn("%s is not a valid file" % sourcefile)
                continue
            taskResult = self._process_single_parse()
            taskResults.append(taskResult)
        self.logger.info(taskResults)

    def _process_single_parse(self):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_single_parse!')
            return
        self.interpreterAccounting.initialize(self.gConfig)
        taskResult = self.interpreterAccounting.doWork(debug=False, tracking=False)
        return taskResult

    def _process_batch_analysize(self):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_batch_analysize!')
            return
        pass

    def _process_single_analysize(self):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_single_analysize!')
            return
        pass

    def _is_file_name_valid(self,fileName):
        assert fileName != None and fileName != NULLSTR, "filename (%s) must not be None or NULL" % fileName
        isFileNameValid = False
        pattern = '年度报告|季度报告'
        if isinstance(pattern, str) and isinstance(fileName, str):
            if pattern != NULLSTR:
                matched = re.search(pattern, fileName)
                if matched is not None:
                    isFileNameValid = True
        return isFileNameValid

    def initialize(self):
        pass

def create_object(gConfig, interpreterDict):
    interpreter = InterpreterNature(gConfig, interpreterDict)
    interpreter.initialize()
    return interpreter