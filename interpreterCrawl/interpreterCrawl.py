#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/20/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterCrawl.py
# @Note    : 用接近自然语言的解释器处理爬虫事务,用于处理财务数据爬取

from ply import lex,yacc
import time
from interpreterCrawl.interpreterBaseClass import *

class InterpreterCrawl(InterpreterBase):
    def __init__(self,gConfig,memberModuleDict):
        super(InterpreterCrawl, self).__init__(gConfig)
        self.crawlFinance = memberModuleDict['crawlfinance']
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


        def p_expression_crawl(p):
            '''expression : SCALE CRAWL'''
            p[0] = p[1] + p[2]


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




    def _is_file_selcted(self,sourcefile):
        assert self.names['公司简称'] != NULLSTR and self.names['报告类型'] != NULLSTR and self.names['报告时间'] != NULLSTR\
            ,"parameter 公司简称,报告类型,报告年度 must not be NULL in 批量处理程序"

        isFileSelected = self._is_matched('|'.join(self.names['公司简称']),sourcefile) \
                         and self._is_matched('|'.join(self.names['报告类型']),sourcefile) \
                         and self._is_matched('|'.join(self.names['报告时间']),sourcefile)
        return isFileSelected



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
        self.names['公司简称'] = NULLSTR
        self.names['报告时间'] = NULLSTR
        self.names['报告类型'] = NULLSTR
        self.names['timelist'] = NULLSTR
        self.names['valuelist'] = NULLSTR


def create_object(gConfig,memberModuleDict):
    interpreter=InterpreterCrawl(gConfig, memberModuleDict)
    interpreter.initialize()
    return interpreter