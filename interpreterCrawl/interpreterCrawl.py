#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/20/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterCrawl.py
# @Note    : 用接近自然语言的解释器处理爬虫事务,用于处理财务数据爬取

from ply import lex,yacc
from interpreterCrawl.interpreterBaseClass import *

class InterpreterCrawl(InterpreterBase):
    def __init__(self,gConfig,memberModuleDict):
        super(InterpreterCrawl, self).__init__(gConfig)
        self.crawlFinance = memberModuleDict['crawlfinance']
        self.crawlStock = memberModuleDict['crawlstock']
        self.checkpointfilename = os.path.join(self.working_directory, gConfig['checkpointfile'])
        self.checkpointIsOn = self.gConfig['checkpointIsOn'.lower()]
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
            self.logger.info("Illegal character '%s'" % t.value[0])
            t.lexer.skip(1)

        # Build the lexer
        self.lexer = lex.lex(outputdir=self.working_directory,reflags=int(re.MULTILINE))

        # dictionary of names_global
        self.names = {}


        def p_statement_expression(p):
            '''statement : statement expression
                         | expression'''
            pass


        def p_expression_crawl(p):
            '''expression : SCALE CRAWL WEBSITE'''
            p[0] = p[1] + p[2] + p[3]
            self.logger.info('Start to crawl finance data from %s'%p[3])
            website = p[3]
            scale = p[1]
            if p[1] != '批量':
                self.logger.warning('the scale %s is not support,now only support scale \'全量\''%p[1])
            self._process_crawl_from_website(website,scale)


        def p_error(p):
            if p:
                print("Syntax error at '%s:%s'" % (p.value,p.type))
                self.logger.error("Syntax error at '%s:%s'" % (p.value, p.type))
            else:
                print("Syntax error at EOF page")
                self.logger.error("Syntax error at EOF page")


        # Build the docparser
        self.parser = yacc.yacc(outputdir=self.working_directory)


    def doWork(self,command,debug=False,tracking=False):
        text = command
        self.parser.parse(text,lexer=self.lexer,debug=debug,tracking=tracking)


    def _get_main_program(self):
        return self._get_text()


    def _process_crawl_from_website(self,website,scale):
        assert website != NULLSTR,"website(%s) is invalid"%website
        if website == '巨潮资讯网':
            self.crawlFinance.initialize(self.gConfig)
            self.crawlFinance.crawl_finance_data(website,scale)
        elif website == '股城网' or website == '东方财富网':
            self.crawlStock.initialize(self.gConfig)
            self.crawlStock.crawl_stock_data(website,scale)


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


    def initialize(self,dicParameter = None):
        if dicParameter is not None:
            self.gConfig.update(dicParameter)


def create_object(gConfig,memberModuleDict):
    interpreter=InterpreterCrawl(gConfig, memberModuleDict)
    interpreter.initialize()
    return interpreter